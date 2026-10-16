# SPDX-License-Identifier: Apache-2.0
"""Regression tests for request-owned rollout state and trajectory collection."""

import types
import unittest
import weakref
from contextlib import nullcontext
from unittest import mock

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.post_training import rollout_hooks, sp_utils


class TestRolloutHooks(unittest.TestCase):
    def setUp(self):
        self.config = types.SimpleNamespace(
            shard_latents_for_sp=lambda **kwargs: (kwargs["latents"], False)
        )
        patcher = mock.patch.object(sp_utils, "get_sp_world_size", return_value=1)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_batch(self, device="cpu"):
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(2, device=device)
        batch = Req(
            sampling_params=SamplingParams(
                prompt="test",
                rollout=True,
                rollout_return_dit_trajectory=True,
                rollout_return_step_indices=[0, 2],
            ),
            latents=torch.zeros(1, 2, device=device),
            scheduler=scheduler,
            generator=torch.Generator(device=device).manual_seed(0),
        )
        rollout_hooks.prepare_rollout_if_enabled(
            batch=batch, pipeline_config=self.config
        )
        rollout_hooks.init_rollout_env_if_enabled(
            batch=batch,
            image_kwargs={},
            pos_cond_kwargs={},
            neg_cond_kwargs={},
            guidance=None,
        )
        self.addCleanup(rollout_hooks.release_rollout_resources, batch)
        return batch

    def test_request_logging_does_not_copy_tensors(self):
        batch = self.make_batch()
        batch.rollout_controllers["visual"].noise_buffer = torch.zeros(16)
        with mock.patch.object(
            torch.Tensor, "__deepcopy__", side_effect=AssertionError("copied tensor")
        ):
            text = str(batch)
        self.assertIn("'rollout_controllers': ['visual']", text)

    def test_failed_denoising_releases_controller_without_garbage_collection(self):
        batch = self.make_batch()
        controller = batch.rollout_controllers["visual"]
        controller.noise_buffer = torch.zeros(16)
        controller_ref = weakref.ref(controller)
        noise_ref = weakref.ref(controller.noise_buffer)
        del controller
        stage = DenoisingStage.__new__(DenoisingStage)
        stage._offload_for_torch_compile_warmup = lambda batch: nullcontext()
        stage._denoise = mock.Mock(side_effect=RuntimeError("denoising failed"))

        with self.assertRaisesRegex(RuntimeError, "denoising failed"):
            stage.forward(batch, types.SimpleNamespace())

        self.assertEqual(batch.rollout_controllers, {})
        self.assertIsNone(batch._rollout_denoising_env_state)
        self.assertIsNone(controller_ref())
        self.assertIsNone(noise_ref())

    def test_failed_finalization_clears_request_state(self):
        batch = self.make_batch()
        with (
            mock.patch.object(
                rollout_hooks,
                "_finalize_rollout_env",
                side_effect=RuntimeError("gather failed"),
            ),
            mock.patch.object(rollout_hooks, "_collect_rollout_log_probs"),
        ):
            with self.assertRaisesRegex(RuntimeError, "gather failed"):
                rollout_hooks.finalize_rollout_if_enabled(
                    batch=batch,
                    latents=batch.latents,
                    num_inference_steps=2,
                    final_timestep=torch.tensor(0.0),
                    pipeline_config=self.config,
                )
        self.assertEqual(batch.rollout_controllers, {})
        self.assertIsNone(batch._rollout_denoising_env_state)

    def test_preparation_clears_stale_state_for_reused_request(self):
        for rollout in (False, True):
            with self.subTest(rollout=rollout):
                batch = self.make_batch()
                previous = weakref.ref(batch.rollout_controllers["visual"])
                batch.rollout = rollout
                rollout_hooks.prepare_rollout_if_enabled(
                    batch=batch, pipeline_config=self.config
                )
                self.assertIsNone(previous())
                self.assertIsNone(batch._rollout_denoising_env_state)
                self.assertEqual(bool(batch.rollout_controllers), rollout)

    def check_trajectory_finalization(self, device):
        batch = self.make_batch(device)
        controller = batch.rollout_controllers["visual"]
        for step, timestep in enumerate(batch.scheduler.timesteps):
            # Recording a device timestep must not call .cpu(), including steps
            # excluded from the latent trajectory by return_step_indices.
            with mock.patch.object(
                torch.Tensor, "cpu", side_effect=AssertionError("per-step CPU copy")
            ):
                controller._maybe_append_dit_trajectory_step(
                    batch.latents, timestep, step
                )
            self.assertEqual(controller.step_timesteps[-1].device, timestep.device)
            controller.append_local_rollout_log_probs(
                torch.zeros(1, device=device), torch.ones(1, device=device)
            )

        # A host terminal timestep must still stack with device loop timesteps.
        rollout_hooks.finalize_rollout_if_enabled(
            batch=batch,
            latents=batch.latents,
            num_inference_steps=2,
            final_timestep=torch.tensor(0.0),
            pipeline_config=self.config,
        )

        data = batch.rollout_trajectory_data
        trajectory = data.dit_trajectory
        self.assertEqual(trajectory.timesteps.device.type, "cpu")
        self.assertEqual(data.rollout_log_probs.device.type, "cpu")
        torch.testing.assert_close(
            trajectory.timesteps,
            torch.cat((batch.scheduler.timesteps.cpu(), torch.zeros(1))),
        )
        self.assertEqual(trajectory.latent_step_indices.tolist(), [0, 2])
        self.assertEqual(trajectory.latents.shape, (1, 2, 2))
        self.assertEqual(batch.rollout_controllers, {})
        self.assertIsNone(batch._rollout_denoising_env_state)

    def test_cpu_trajectory_finalization(self):
        self.check_trajectory_finalization("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_trajectory_finalization(self):
        self.check_trajectory_finalization("cuda")


if __name__ == "__main__":
    unittest.main()
