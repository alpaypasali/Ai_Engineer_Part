import argparse
import json
from pathlib import Path

from src.serving import load_model, predict_purchase


def main():


    parser = argparse.ArgumentParser(
        description="ShopperIntent Purchase Prediction CLI"
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to a JSON file containing session data."
    )

    args = parser.parse_args()

    try:
   
        with open(args.input_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)

  
        if not isinstance(session_data, dict):
            raise TypeError(
                "Session data must be a JSON object."
            )

     
        model, preprocessor, config = load_model()

      
        result = predict_purchase(
            session_data=session_data,
            model=model,
            preprocessor=preprocessor,
            config=config
        )

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError
    ) as error:

        parser.exit(
            status=1,
            message=f"Error: {error}\n"
        )

    print("\n================================")
    print("       SHOPPER INTENT AI")
    print("================================\n")

    print(
        f"Purchase Probability : "
        f"{result['purchase_probability']:.4f}"
    )

    print(
        f"Predicted Revenue    : "
        f"{result['predicted_revenue']}"
    )

    print(
        f"Campaign Candidate   : "
        f"{'Yes' if result['target_campaign'] else 'No'}"
    )

    print(
        f"Decision Threshold   : "
        f"{result['threshold']:.2f}"
    )

    print(
        f"Model Version        : "
        f"{result['model_version']}"
    )

    print("\n================================")


if __name__ == "__main__":
    main()