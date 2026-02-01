import argparse
from agents.trainer import RLTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta un modelo entrenado de RL en el entorno MuJoCo"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path al modelo entrenado (.zip)"
    )

    parser.add_argument(
        "--xml-file",
        type=str,
        required=True,
        help="Path al archivo XML de MuJoCo"
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="PPO",
        choices=["PPO", "SAC", "A2C"],
        help="Tipo de agente"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Cantidad de episodios a ejecutar"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Máximo de pasos por episodio"
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Usar política determinista"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para reproducibilidad"
    )

    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human"],
        help="Modo de render (solo human en predict)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    trainer = RLTrainer(
        agent_type=args.agent,
        xml_file=args.xml_file,
        render_mode=args.render_mode,
        max_steps=args.max_steps,
        seed=args.seed,
        create_run_dir=False
    )

    trainer.predict(
        model_path=args.model_path,
        episodes=args.episodes,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()