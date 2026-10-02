"""Train an RL-token extractor from frozen SmolVLA image representations.

Example:
    lerobot-rlt-train --dataset.repo_id=... --policy.path=lerobot/smolvla_base
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets import LeRobotDataset
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.rl_token import RLTokenExtractor
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


@torch.no_grad()
def extract_image_tokens(
    policy: SmolVLAPolicy,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run the frozen VLM prefix and return its final image-patch tokens."""
    if policy.config.add_image_special_tokens:
        raise ValueError(
            "Image-only slicing requires add_image_special_tokens=False so image patches are contiguous."
        )

    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks, num_image_tokens = (
        policy.model.embed_prefix(
            images,
            image_masks,
            lang_tokens,
            lang_masks,
            state=state,
            return_image_token_count=True,
        )
    )

    prefix_attention_mask = make_att_2d_masks(
        prefix_pad_masks,
        prefix_att_masks,
    )
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    outputs, _ = policy.model.vlm_with_expert.forward(
        attention_mask=prefix_attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
    )

    # add_image_special_tokens=False guarantees patches occupy the prefix.
    image_tokens = outputs[0][:, :num_image_tokens]
    if image_tokens.shape[1] != num_image_tokens:
        raise RuntimeError("The extracted image-token count does not match the prefix layout.")
    return image_tokens


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    if policy.config.add_image_special_tokens:
        raise ValueError("Set policy.add_image_special_tokens=False for image-patch slicing.")

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    hidden_dim = policy.model.vlm_with_expert.config.text_config.hidden_size
    extractor = RLTokenExtractor(
        dim=hidden_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        extractor.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    step = 0
    while step < args.steps:
        for batch in dataloader:
            batch = preprocessor(batch)
            z = extract_image_tokens(policy, batch).float()

            _, _, loss = extractor(z)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), args.max_grad_norm)
            optimizer.step()

            if step % args.log_freq == 0:
                print(f"step={step} loss={loss.item():.6f}")

            step += 1
            if step >= args.steps:
                break

    torch.save(
        {
            "model": extractor.state_dict(),
            "hidden_dim": hidden_dim,
            "image_patch_count": z.shape[1],
            "step": step,
        },
        output_dir / "rl_token_extractor.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", required=True)
    parser.add_argument("--dataset.root", dest="dataset_root", default=None)
    parser.add_argument("--policy.path", dest="policy_path", default="lerobot/smolvla_base")
    parser.add_argument("--output_dir", default="outputs/rl_token")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--encoder_depth", type=int, default=2)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
