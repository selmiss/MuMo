#!/usr/bin/env python3
"""
Calculate Mean Absolute Error (MAE) between two columns of a CSV file.

Usage:
    python calculate_mae.py --csv_file data.csv --col1 actual --col2 predicted
    python calculate_mae.py --csv_file data.csv --col1 actual --col2 predicted --output results.txt
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path


def calculate_mae(actual, predicted):
    """
    Calculate Mean Absolute Error between two arrays.

    Args:
        actual: Array of actual values
        predicted: Array of predicted values

    Returns:
        float: Mean Absolute Error
    """
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have the same length")

    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]

    if len(actual_clean) == 0:
        raise ValueError("No valid data points after removing NaN values")

    mae = np.mean(np.abs(actual_clean - predicted_clean))
    return mae


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Mean Absolute Error (MAE) between two columns of a CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--csv_file", type=str, required=True, help="Path to the CSV file"
    )

    parser.add_argument(
        "--col1",
        type=str,
        required=True,
        help="Name of the first column (actual values)",
    )

    parser.add_argument(
        "--col2",
        type=str,
        required=True,
        help="Name of the second column (predicted values)",
    )

    parser.add_argument(
        "--output", type=str, help="Output file to save results (optional)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about the calculation",
    )

    args = parser.parse_args()

    # Check if CSV file exists
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file '{args.csv_file}' not found.")
        sys.exit(1)

    try:
        # Read CSV file
        print(f"Reading CSV file: {args.csv_file}")
        df = pd.read_csv(args.csv_file)

        # Check if columns exist
        if args.col1 not in df.columns:
            print(f"Error: Column '{args.col1}' not found in CSV file.")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)

        if args.col2 not in df.columns:
            print(f"Error: Column '{args.col2}' not found in CSV file.")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)

        # Extract columns
        actual = df[args.col1].values
        predicted = df[args.col2].values

        if args.verbose:
            print(f"Column '{args.col1}' (actual): {len(actual)} values")
            print(f"Column '{args.col2}' (predicted): {len(predicted)} values")
            print(
                f"Actual values range: {np.nanmin(actual):.4f} to {np.nanmax(actual):.4f}"
            )
            print(
                f"Predicted values range: {np.nanmin(predicted):.4f} to {np.nanmax(predicted):.4f}"
            )

        # Calculate MAE
        mae = calculate_mae(actual, predicted)

        # Print results
        print(f"\n{'='*50}")
        print(f"MAE Calculation Results")
        print(f"{'='*50}")
        print(f"File: {args.csv_file}")
        print(f"Actual column: {args.col1}")
        print(f"Predicted column: {args.col2}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"{'='*50}")

        # Save to output file if specified
        if args.output:
            with open(args.output, "w") as f:
                f.write(f"MAE Calculation Results\n")
                f.write(f"{'='*50}\n")
                f.write(f"File: {args.csv_file}\n")
                f.write(f"Actual column: {args.col1}\n")
                f.write(f"Predicted column: {args.col2}\n")
                f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
                f.write(f"{'='*50}\n")
            print(f"Results saved to: {args.output}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
