import random


def split_file(input_file, split_sizes, split_names):

    assert len(split_sizes) == len(split_names)

    with open(input_file, 'r') as file:
        entries = file.readlines()

    assert sum(split_sizes) == len(entries)

    entries = [entry.strip() for entry in entries]

    random.shuffle(entries)

    start_index = 0

    # create the output files based on the split sizes
    for i, size in enumerate(split_sizes):
        end_index = start_index + size
        split_entries = entries[start_index:end_index]

        output_file = f'{split_names[i]}.txt'
        with open(output_file, 'w') as file:
            for entry in split_entries:
                file.write(entry + '\n')

        start_index = end_index

    print("Split complete! Created the following files:")
    for i in range(len(split_sizes)):
        print(f'{split_names[i]}.txt')


random.seed(0)
input_file = '../data/splits/scannetv2_living.txt'
split_sizes = [168, 45, 11]
split_names = ['scannetv2_living_train', 'scannetv2_living_val', 'scannetv2_living_test']
split_file(input_file, split_sizes, split_names)