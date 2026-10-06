"""GAN + VAE loss terms combined into VITS's training objective."""
import torch


def feature_loss(fmap_r, fmap_g) -> torch.Tensor:
    """L1 distance between real/generated discriminator feature maps
    (encourages the generator to match real audio's internal discriminator
    activations, not just fool the final real/fake output)."""
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl.float().detach() - gl.float()))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """LSGAN-style discriminator loss: real -> 1, generated -> 0."""
    loss = 0.0
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr.float()) ** 2)
        g_loss = torch.mean(dg.float() ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """LSGAN-style generator loss: wants the discriminator to output 1."""
    loss = 0.0
    gen_losses = []
    for dg in disc_outputs:
        l_dg = torch.mean((1 - dg.float()) ** 2)
        gen_losses.append(l_dg)
        loss += l_dg
    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask) -> torch.Tensor:
    """KL(q(z|spec) || p(z|text)), masked and averaged over valid frames."""
    z_p, logs_q, m_p, logs_p, z_mask = (
        t.float() for t in (z_p, logs_q, m_p, logs_p, z_mask)
    )
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.sum(z_mask)
