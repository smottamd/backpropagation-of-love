import os
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# BACKPROPAGATION DO AMOR
# Uma rede neural aprendendo a desenhar um coração aos poucos.
# ============================================================


GIF_TAMANHO = 600
IMAGEM_TAMANHO = 1080

MAX_FRAMES_LINKEDIN = 96
JANELA_RODADAS_VISIVEIS = 2000

RODADAS = 20_000_000
LEARNING_RATE = 1.0
TEMPO_MAXIMO_SEGUNDOS = 60 * 60

OUTPUT_DIR = "saida_amor"


def sigma(z):
    z = np.clip(z, -60, 60)
    return 1 / (1 + np.exp(-z))


def criar_dados_coracao(n=180):
    t = np.linspace(0, 2 * np.pi, n)

    x = 16 * np.sin(t) ** 3
    y = (
        13 * np.cos(t)
        - 5 * np.cos(2 * t)
        - 2 * np.cos(3 * t)
        - np.cos(4 * t)
    )

    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    entrada = np.linspace(0, 1, n).reshape(1, n)
    saida = np.vstack([x, y])

    return entrada, saida


def iniciar_rede(n1=12, n2=16, seed=42):
    global W1, W2, W3, b1, b2, b3

    rng = np.random.default_rng(seed)

    W1 = rng.standard_normal((n1, 1)) / 2
    W2 = rng.standard_normal((n2, n1)) / 2
    W3 = rng.standard_normal((2, n2)) / 2

    b1 = rng.standard_normal((n1, 1)) / 2
    b2 = rng.standard_normal((n2, 1)) / 2
    b3 = rng.standard_normal((2, 1)) / 2


def rede(a0):
    z1 = W1 @ a0 + b1
    a1 = sigma(z1)

    z2 = W2 @ a1 + b2
    a2 = sigma(z2)

    z3 = W3 @ a2 + b3
    a3 = sigma(z3)

    return a0, z1, a1, z2, a2, z3, a3


def custo(x, y):
    predicao = rede(x)[-1]
    return np.mean((predicao - y) ** 2)


def gradientes(x, y):
    a0, z1, a1, z2, a2, z3, a3 = rede(x)
    m = x.shape[1]

    delta3 = 2 * (a3 - y) * a3 * (1 - a3)

    dW3 = delta3 @ a2.T / m
    db3 = np.sum(delta3, axis=1, keepdims=True) / m

    delta2 = (W3.T @ delta3) * a2 * (1 - a2)

    dW2 = delta2 @ a1.T / m
    db2 = np.sum(delta2, axis=1, keepdims=True) / m

    delta1 = (W2.T @ delta2) * a1 * (1 - a1)

    dW1 = delta1 @ a0.T / m
    db1 = np.sum(delta1, axis=1, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3


def passo_treino(x, y, learning_rate=1.0):
    global W1, W2, W3, b1, b2, b3

    dW1, db1, dW2, db2, dW3, db3 = gradientes(x, y)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3


def abrir_pasta(path):
    path = Path(path).resolve()

    try:
        if os.name == "nt":
            os.startfile(str(path))
    except Exception:
        pass


def criar_rodadas_snapshot(rounds, max_frames):
    comeco_lento = list(range(0, 41))
    comeco_lento += [
        50,
        60,
        75,
        90,
        110,
        140,
        180,
        230,
        300,
        400,
        550,
        750,
        1000,
        1400,
        2000,
        3000,
        4500,
        6500,
        9000,
        13000,
        18000,
        25000,
        35000,
        50000,
        75000,
        100000,
    ]

    rodadas = [r for r in comeco_lento if r <= rounds]
    restantes = max_frames - len(set(rodadas))

    if restantes > 0 and rounds > 100000:
        aceleradas = np.geomspace(150000, rounds, restantes)
        rodadas += [int(round(r)) for r in aceleradas]

    rodadas.append(rounds)
    rodadas = sorted(set(max(0, min(rounds, r)) for r in rodadas))

    if len(rodadas) > max_frames:
        rodadas = rodadas[: max_frames - 1] + [rounds]

    return set(rodadas)


def salvar_tracados(
    y,
    snapshots,
    output_path,
    titulo,
    tamanho=600,
):
    fig = plt.figure(figsize=(tamanho / 100, tamanho / 100), dpi=100)
    ax = fig.add_axes([0.055, 0.105, 0.89, 0.78])

    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")

    ax.plot(
        y[0],
        y[1],
        linewidth=2.2,
        alpha=0.16,
        color="white",
        label="coração real",
    )

    ultima_rodada, _, ultimo_custo = snapshots[-1]

    snapshots_visiveis = [
        item
        for item in snapshots
        if ultima_rodada - item[0] <= JANELA_RODADAS_VISIVEIS
    ]

    total = len(snapshots_visiveis)

    for idx, (rodada, predicao, cst) in enumerate(snapshots_visiveis, start=1):
        alpha = 0.08 + 0.78 * (idx / total)
        linewidth = 0.35 + 1.35 * (idx / total)

        ax.plot(
            predicao[0],
            predicao[1],
            linewidth=linewidth,
            alpha=alpha,
            color="#ff4f8b",
        )

    texto_rodada = f"rodada {ultima_rodada:,}".replace(",", ".")

    ax.set_title(
        f"{titulo}\n{texto_rodada} | custo {ultimo_custo:.6f}",
        fontsize=15 if tamanho <= 700 else 24,
        color="white",
        pad=18,
    )

    fig.text(
        0.5,
        0.055,
        "todos os gradientes apontam para você",
        ha="center",
        va="center",
        fontsize=11 if tamanho <= 700 else 18,
        color="white",
        alpha=0.85,
    )

    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(output_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def criar_gif(frame_paths, output_path, duration=60):
    try:
        from PIL import Image
    except ImportError:
        print("Pillow não está instalado. Para criar GIF: python -m pip install pillow")
        return

    if not frame_paths:
        return

    images = [Image.open(path) for path in frame_paths]

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )

    for image in images:
        image.close()


def adicionar_snapshot(snapshots, x, y, rodada, max_frames):
    predicao = rede(x)[-1].copy()
    cst = custo(x, y)

    if len(snapshots) < max_frames:
        snapshots.append((rodada, predicao, cst))
    else:
        snapshots[-1] = (rodada, predicao, cst)

    return cst


def salvar_snapshot(
    y,
    snapshots,
    frame_paths,
    output_dir,
    frames_dir,
    titulo,
):
    imagem_principal = output_dir / "amor_tracados.png"

    salvar_tracados(
        y=y,
        snapshots=snapshots,
        output_path=imagem_principal,
        titulo="Backpropagation do Amor",
        tamanho=IMAGEM_TAMANHO,
    )

    frame_path = frames_dir / f"frame_{len(snapshots):03d}.png"

    salvar_tracados(
        y=y,
        snapshots=snapshots,
        output_path=frame_path,
        titulo=titulo,
        tamanho=GIF_TAMANHO,
    )

    frame_paths.append(frame_path)


def renderizar_resultado(y, snapshots, output_dir, frames_dir):
    frame_paths = []

    for frame_antigo in frames_dir.glob("frame_*.png"):
        frame_antigo.unlink()

    for idx in range(1, len(snapshots) + 1):
        salvar_snapshot(
            y=y,
            snapshots=snapshots[:idx],
            frame_paths=frame_paths,
            output_dir=output_dir,
            frames_dir=frames_dir,
            titulo="Backpropagation do Amor",
        )

    return frame_paths


def atualizar_preview(y, snapshots, output_dir):
    salvar_tracados(
        y=y,
        snapshots=snapshots,
        output_path=output_dir / "amor_tracados.png",
        titulo="Backpropagation do Amor",
        tamanho=IMAGEM_TAMANHO,
    )


def treinar(
    x,
    y,
    rounds=RODADAS,
    learning_rate=LEARNING_RATE,
    output_dir=OUTPUT_DIR,
    tempo_maximo_segundos=TEMPO_MAXIMO_SEGUNDOS,
    max_frames=MAX_FRAMES_LINKEDIN,
):
    output_dir = Path.cwd() / output_dir
    frames_dir = output_dir / "frames"

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    abrir_pasta(output_dir)

    snapshots = []
    rodadas_snapshot = criar_rodadas_snapshot(rounds, max_frames)
    inicio = time.time()

    cst = adicionar_snapshot(
        snapshots=snapshots,
        x=x,
        y=y,
        rodada=0,
        max_frames=max_frames,
    )

    atualizar_preview(y, snapshots, output_dir)

    print(f"Rodada {0:>9} | custo = {cst:.6f} | imagem atualizada")

    for i in range(1, rounds + 1):
        passo_treino(x, y, learning_rate=learning_rate)
        tempo_passado = time.time() - inicio

        salvar_por_rodada = i in rodadas_snapshot
        chegou_no_fim = i == rounds or tempo_passado >= tempo_maximo_segundos

        if salvar_por_rodada or chegou_no_fim:
            cst = adicionar_snapshot(
                snapshots=snapshots,
                x=x,
                y=y,
                rodada=i,
                max_frames=max_frames,
            )

            atualizar_preview(y, snapshots, output_dir)

            print(
                f"Rodada {i:>9} | custo = {cst:.6f} | "
                f"tempo = {tempo_passado / 60:.1f} min | "
                f"snapshots = {len(snapshots)} | imagem atualizada"
            )

        if tempo_passado >= tempo_maximo_segundos:
            break

    print()
    print("Renderizando PNG e GIF...")

    frame_paths = renderizar_resultado(
        y=y,
        snapshots=snapshots,
        output_dir=output_dir,
        frames_dir=frames_dir,
    )

    gif_path = output_dir / "amor_animado.gif"
    criar_gif(frame_paths, gif_path)

    print()
    print("PRONTO.")
    print(f"Imagem principal: {output_dir / 'amor_tracados.png'}")
    print(f"GIF animado:       {gif_path}")
    print(f"Frames:            {frames_dir}")
    print(f"Total de frames:   {len(frame_paths)}")
    print(f"Pixels do GIF:     {GIF_TAMANHO * GIF_TAMANHO * len(frame_paths):,}")
    print(f"Rodadas treinadas: {snapshots[-1][0]}")
    print(f"Custo final:       {snapshots[-1][2]:.6f}")

    abrir_pasta(output_dir)


def main():
    x, y = criar_dados_coracao(n=180)

    iniciar_rede(n1=12, n2=16, seed=42)

    print("Rede iniciada.")
    print(f"Custo inicial: {custo(x, y):.6f}")
    print()
    print("A pasta de saída será aberta agora.")
    print("Não aperte Ctrl+C. As imagens vão aparecer aos poucos em saida_amor.")
    print()

    treinar(
        x=x,
        y=y,
    )


if __name__ == "__main__":
    main()
