import json
import numpy as np
from pathlib import Path
import argparse

def load_epochs(train_samples_path, steps_per_epoch, max_epoch):
    epochs = []
    prompts = set()
    current_epoch = []

    for step in range(1, steps_per_epoch * max_epoch + 1):
        with open(train_samples_path / f'step_{step}.json') as f:
            step_data = json.load(f)
            current_epoch.extend(step_data)

        if step % steps_per_epoch == 0:
            prompts.update(sample['prompt'] for sample in current_epoch)
            epochs.append(current_epoch)
            current_epoch = []

    return epochs, prompts

'''
    return 
    {
        "prompt1": accuracy_value1,
        "prompt2": accuracy_value2,
        "prompt3": accuracy_value3,
        ...
    }
'''
def calculate_accuracy(epoch_data, prompt_set):
    # 构造一个dict key: prompt value: [correct_count, total_count]
    accuracies = {prompt: [0, 0] for prompt in prompt_set}
    for sample in epoch_data:
        # strip() 移除开头结尾两部分
        prompt = sample['prompt'].strip()
        accuracies[prompt][1] += 1
        if sample['reward'] == 1:
            accuracies[prompt][0] += 1
    # total 不存在尝试
    return {prompt: correct / total if total else -1 for
            prompt, (correct, total) in accuracies.items()}

def process_accuracy_sequence(prompt_accuracies, max_epochs):
    for accuracy_sequence in prompt_accuracies.values():
        for i in range(len(accuracy_sequence) - 1):
            # backward fill 技术
            if accuracy_sequence[i] == -1 and accuracy_sequence[i + 1] != -1:
                accuracy_sequence[i] = accuracy_sequence[i + 1]
    # filter valid sequence, sequence must not have -1, or squence will not be used
    valid_sequence = [(prompt, sequence) for prompt, sequence in prompt_accuracies.items()
                      if -1 not in sequence[:max_epochs]]
    if not valid_sequence:
        return [], [], []

    prompts, sequences = zip(*valid_sequence)
    sequence = [seq[:max_epochs] for seq in sequences]
    mean_sequence = np.mean(sequence, axis=0)

    return prompts, sequence, mean_sequence

def calculate_similarity_score(sequence, baseline_sequence):
    # squared_diff_sum = sum((r_i - r_avg)**2 for r_i, r_avg in zip(sample_rewards, epoch_avg_rewards))
    # max_diff = sum((1 - r_avg)**2 for r_avg in epoch_avg_rewards)
    squared_diff_sum = sum((acc - baseline)**2 for acc, baseline in zip(sequence, baseline_sequence))
    max_diff_sum = sum((1 - baseline)**2 for baseline in baseline_sequence)
    return 1 - squared_diff_sum / max_diff_sum

def parse_args():
    parser = argparse.ArgumentParser(description='Process training data and filter prompts')
    parser.add_argument('--train_samples_path', type=str, required=True,
                        help='Path to the training samples directory')
    parser.add_argument('--original_prompts_path', type=str, required=True,
                        help='Path to the original prompts json file')
    parser.add_argument('--output_path', type=str, default='math.sub.average_filtered.json',
                        help='Path for output filtered data')
    parser.add_argument('--steps_per_epoch', type=int, default=8,
                        help='Number of steps that constitute one epoch')
    parser.add_argument('--max_epochs', type=int, default=21,
                        help='Maximum number of epochs to consider')
    parser.add_argument('--similarity_threshold', type=float, default=0.2,
                        help='Minimum similarity score threshold for selecting prompts')
    return parser.parse_args()

def main():
    args = parse_args()
    train_samples_path = Path(args.train_samples_path)
    epochs, prompts = load_epochs(train_samples_path, args.steps_per_epoch, args.max_epochs)

    epoch_accuracies = [calculate_accuracy(epoch, prompts) for epoch in epochs]

    # collect accuracy sequence for each prompt
    prompt_accuracies = {prompt: [epoch[prompt] for epoch in epoch_accuracies]
                        for prompt in prompts}

    valid_accuracies, accuracy_sequences, baseline_sequence = process_accuracy_sequence(
        prompt_accuracies, args.max_epochs)

    prompt_score = {prompt: calculate_similarity_score(sequence, baseline_sequence)
                    for prompt, sequence in zip(accuracy_sequences, baseline_sequence)}

    selected_prompt = {prompt for prompt, score in prompt_score.items()
                       if score >= args.similarity_threshold}

    with open(args.original_prompts_path) as f:
        original_data = json.load(f)

    filtered_data = [sample for sample in original_data if sample['prompt'] in selected_prompt]
    with open(args.output_path, 'w') as f:
        json.dump(filtered_data, f)

    print(f"Selected {len(filtered_data)} prompts")

if __name__ == '__main__':
    main()