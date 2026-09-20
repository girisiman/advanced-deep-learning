import torch
import torch.nn as nn

def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """Average cross-entropy over non-pad positions.

    Args:
        logits: Model scores, shape (batch_size, time_steps, vocab_size).
        labels: True next-word ids, shape (batch_size, time_steps).
        pad_id: Token id that must not enter the loss.

    Returns:
        A scalar tensor (the mean loss).
    """
    vocab_size = logits.size(dim=-1)
    # (batch * time, vocab) and (batch * time,)
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    loss = loss_fn(input=logits_flat, target=labels_flat)
    return loss

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    pad_id: int,
    device: torch.device | None = None,
    max_batches: int | None = None,
    grad_clip: float = 1.0,
) -> float:
    """Train the model on the loader once.

    Args:
        model: NextWordModel.
        loader: Train DataLoader from dataset.py.
        optimizer: Usually Adam.
        pad_id: Id ignored by the loss.
        device: cpu or cuda. If None, use the model's device.
        max_batches: If set, stop after this many batches (for tests).
        grad_clip: Max gradient norm. 0 disables clipping.

    Returns:
        Mean training loss over the batches that were used.
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model.forward(input_ids=input_ids)
        loss = compute_loss(logits=logits, labels=labels, pad_id=pad_id)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    return mean_loss

def evaluate(
    model: nn.Module,
    loader,
    pad_id: int,
    device: torch.device | None = None,
    max_batches: int | None = None,
) -> dict:
    """Compute loss, perplexity and accuracy on a loader.

    Args:
        model: NextWordModel.
        loader: Val or test DataLoader.
        pad_id: Id ignored by the loss and by accuracy.
        device: cpu or cuda. If None, use the model's device.
        max_batches: If set, stop early (for notebook tests).

    Returns:
        Dict with keys loss, ppl, top1, top5, acc_at_k, n_tokens.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    n_batches = 0
    n_tokens = 0
    n_top1 = 0
    n_top5 = 0
    acc_at_k = {4: [0, 0], 8: [0, 0], 12: [0, 0], 16: [0, 0]}

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"]

            logits = model.forward(input_ids=input_ids)
            loss = compute_loss(logits=logits, labels=labels, pad_id=pad_id)
            total_loss += float(loss.item())
            n_batches += 1

            # (batch, time)
            pred_top1 = torch.argmax(logits, dim=-1)
            pred_top5 = torch.topk(logits, k=5, dim=-1).indices
            real = labels != pad_id

            n_tokens += int(real.sum().item())
            n_top1 += int(((pred_top1 == labels) & real).sum().item())
            n_top5 += int(
                ((pred_top5 == labels.unsqueeze(-1)).any(dim=-1) & real)
                .sum()
                .item()
            )

            # last column is the next-word target for this window
            last_pred = pred_top1[:, -1]
            last_gold = labels[:, -1]
            for row in range(last_gold.size(0)):
                k = int(lengths[row].item())
                if k not in acc_at_k:
                    continue
                acc_at_k[k][1] += 1
                if int(last_pred[row].item()) == int(last_gold[row].item()):
                    acc_at_k[k][0] += 1

    mean_loss = total_loss / max(n_batches, 1)
    metrics = {
        "loss": mean_loss,
        "ppl": float(torch.exp(torch.tensor(mean_loss))),
        "top1": n_top1 / max(n_tokens, 1),
        "top5": n_top5 / max(n_tokens, 1),
        "n_tokens": n_tokens,
        "acc_at_k": {
            k: (hits / max(total, 1)) for k, (hits, total) in acc_at_k.items()
        },
    }
    return metrics

def generate(
    model: nn.Module,
    vocab,
    seed_tokens: list[str],
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    device: torch.device | None = None,
) -> list[str]:
    """Generate words after a seed phrase.

    Args:
        model: Trained NextWordModel.
        vocab: Vocab object from dataset.py (needs encode and decode).
        seed_tokens: Starting words, already split, e.g. ["first", "citizen", ":"].
        max_new_tokens: How many new words to append.
        temperature: 0 means greedy argmax. >0 means sampling.
        device: cpu or cuda. If None, use the model's device.

    Returns:
        seed tokens plus the generated tokens.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    token_ids = vocab.encode(tokens=seed_tokens)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            logits = model.forward(input_ids=input_ids)
            next_logits = logits[0, -1, :]

            if temperature <= 0:
                next_id = int(torch.argmax(next_logits).item())
            else:
                next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            token_ids.append(next_id)
            if vocab.itos[next_id] == "<eos>":
                break

    return vocab.decode(ids=token_ids)