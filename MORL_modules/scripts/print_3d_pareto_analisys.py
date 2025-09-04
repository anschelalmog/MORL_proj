import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# Script metadata
# Date: 2025-08-08 09:43:48
# User: RotemShezaf

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'MORL_modules/run_algos'))
from hyper_morl_utils import aditional_analysis_from_save_results_3d_hv


def main():
    try:
        # Define episodes to analyze
        episodes = [0, 2, 10, 20, 30, 40, 50, 56]

        # Base path for results
        base_path = "/home/rotem.shezaf/MORL_proj/MORL_modules/results/Hyper-MORL/4-objdefault/0"

        # Output directory as specified
        output_path = os.path.join(base_path, "hv_over episodes")
        os.makedirs(output_path, exist_ok=True)

        # Store results for each episode
        results = {}

        # Loop through each episode
        for episode in episodes:
            # Construct the file path for this episode
            episode_path = os.path.join(base_path, str(episode))

            print(f"Analyzing episode {episode} from path: {episode_path}")

            # Get time and hypervolume data for this episode
            time, hv_3d = aditional_analysis_from_save_results_3d_hv(episode_path)

            # Store the results
            results[episode] = {
                'time': time,
                'hv_3d': hv_3d
            }

            # Print summary for this episode
            print(f"  Episode {episode}: Hypervolume = {hv_3d:.6f}")

        # Create a plot comparing hypervolume across episodes
        plt.figure(figsize=(12, 7))
        episode_values = []
        hv_values = []
        time_values = []
        for episode in sorted(episodes):
            if episode in results:
                # Use the last value in hv_3d as the final hypervolume for this episode
                episode_values.append(episode)
                hv_values.append(results[episode]['hv_3d'])
                time_values.append(results[episode]['time'] )

        plt.plot(episode_values, hv_values, marker='o', linestyle='-', color='blue')
        plt.scatter(episode_values, hv_values, color='red', s=100, zorder=5)

        for i, episode in enumerate(episode_values):
            plt.annotate(f'{episode}',
                         (episode, hv_values[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

        plt.title('3D Hypervolume Progress Across Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Final 3D Hypervolume')
        plt.grid(True)
        plt.xticks(episode_values)

        # Save the figure to the specified location
        plot_path = os.path.join(output_path, 'hypervolume_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")

        # Save the raw data to the same location
        data_path = os.path.join(output_path, 'hypervolume_data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Data saved to {data_path}")

        # Create a CSV file with the results
        csv_path = os.path.join(output_path, 'hypervolume_summary.csv')
        with open(csv_path, 'w') as f:
            f.write("Episode,Hypervolume,Improvement from Ep 0,Improvement Percentage\n")

            baseline = results[0]['hv_3d'] if 0 in results else None

            for episode in sorted(episodes):
                if episode in results:
                    hv = results[episode]['hv_3d']
                    if baseline is not None and episode > 0:
                        improvement = hv - baseline
                        percent = (improvement / baseline) * 100 if baseline != 0 else float('inf')
                        f.write(f"{episode},{hv},{improvement},{percent}\n")
                    else:
                        f.write(f"{episode},{hv},0,0\n")

        print(f"CSV summary saved to {csv_path}")

        # Print summary table
        print("\nHypervolume Summary Table:")
        print("-" * 40)
        print("Episode | Final Hypervolume | Improvement from Ep 0")
        print("-" * 40)

        baseline = results[0]['hv_3d'] if 0 in results else None

        for episode in sorted(episodes):
            if episode in results:
                hv = results[episode]['hv_3d']
                if baseline is not None and episode > 0:
                    improvement = hv - baseline
                    percent = (improvement / baseline) * 100 if baseline != 0 else float('inf')
                    print(f"{episode:7d} | {hv:17.6f} | {improvement:+.6f} ({percent:+.2f}%)")
                else:
                    print(f"{episode:7d} | {hv:17.6f} | {'N/A':>20}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()