import numpy as np
from sklearn.metrics import adjusted_rand_score
from os import walk
from statistics import mean, median, stdev
import argparse


def parse_rttm(rttm_file):
    """Parse an RTTM file and return a list of (start_time, end_time, speaker_id)."""
    segments = []
    with open(rttm_file, 'r') as f:
        for line in f:
            if line.startswith('SPEAKER'):
                parts = line.strip().split()
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]
                end_time = start_time + duration
                segments.append((start_time, end_time, speaker_id))
    return segments


def discretize_segments(segments, duration, bin_size=0.1):
    """
    Convert RTTM segments into per-bin speaker labels.
    Args:
        segments: List of (start_time, end_time, speaker_id).
        duration: Total duration of the audio (in seconds).
        bin_size: Size of each time bin (in seconds).
    Returns:
        List of speaker labels for each time bin.
    """
    num_bins = int(np.ceil(duration / bin_size))
    labels = [None] * num_bins  # Initialize with None for non-speech or unassigned

    # Map speaker IDs to integers for ARI
    speaker_map = {sid: i for i, sid in enumerate(set(seg[2] for seg in segments))}

    for start, end, speaker_id in segments:
        start_bin = int(start / bin_size)
        end_bin = int(end / bin_size)
        for i in range(max(0, start_bin), min(num_bins, end_bin + 1)):
            labels[i] = speaker_map[speaker_id]

    # Replace None with a unique label for non-speech (if needed)
    non_speech_label = len(speaker_map)
    labels = [non_speech_label if label is None else label for label in labels]

    return labels


def calculate_ari_from_rttm(ground_truth_rttm, predicted_rttm, bin_size=0.1):
    """
    Calculate ARI from ground truth and predicted RTTM files.
    Args:
        ground_truth_rttm: Path to ground truth RTTM file.
        predicted_rttm: Path to predicted RTTM file.
        bin_size: Time bin size for discretization (in seconds).
    Returns:
        float: ARI score.
    """
    # Parse RTTM files
    gt_segments = parse_rttm(ground_truth_rttm)
    pred_segments = parse_rttm(predicted_rttm)

    # Estimate audio duration (max end time from both files)
    gt_max_time = max(seg[1] for seg in gt_segments) if gt_segments else 0
    pred_max_time = max(seg[1] for seg in pred_segments) if pred_segments else 0
    duration = max(gt_max_time, pred_max_time)

    if duration == 0:
        return 0.0  # No segments to compare

    # Discretize into time bins
    gt_labels = discretize_segments(gt_segments, duration, bin_size)
    pred_labels = discretize_segments(pred_segments, duration, bin_size)

    # Calculate ARI
    ari = adjusted_rand_score(gt_labels, pred_labels)
    return ari


def main():
    # _, _, files = next(walk(r"C:\Users\lenovo\AppData\Roaming\JetBrains\PyCharmCE2021.3\scratches\rttm_git_repo\dev"))
    _, _, files = next(walk(r"C:\Users\lenovo\AppData\Roaming\JetBrains\PyCharmCE2021.3\scratches\rttm_git_repo\test"))
    ari_scores = []
    for file in files:

        # ground_truth = r"C:\Users\lenovo\AppData\Roaming\JetBrains\PyCharmCE2021.3\scratches\rttm_git_repo\dev\{}".format(file)
        ground_truth = r"C:\Users\lenovo\AppData\Roaming\JetBrains\PyCharmCE2021.3\scratches\rttm_git_repo\test\{}".format(file)
        # predicted = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_dev_rttm_desraj\{}".format(file)
        # predicted = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_dev_rttm_aish\{}".format(file)
        # predicted = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_test_rttm_desraj\{}".format(file)
        predicted = r"C:\Users\lenovo\PycharmProjects\pythonProject\vox_test_rttm_aish\{}".format(file)
        ari_score = calculate_ari_from_rttm(ground_truth, predicted)
        print(f"Adjusted Rand Index: {ari_score:.4f}")
        ari_scores.append(ari_score)

    avg_ari = mean(ari_scores)
    print(f"Average ARI: {avg_ari:.4f}")


if __name__ == "__main__":
    main()