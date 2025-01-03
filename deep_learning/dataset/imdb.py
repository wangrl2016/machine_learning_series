import os
import pathlib

if __name__ == '__main__':
    base_url = '/Users/admin/Downloads/aclImdb'
    train_postive_dir = 'train/pos'
    train_negative_dir = 'train/neg'
    test_positive_dir = 'test/pos'
    test_negative_dir = 'test/neg'
    dataset_dir_list = [train_postive_dir, train_negative_dir,
                        test_positive_dir, test_negative_dir]

    train_postive_list = []
    train_negative_list = []
    test_postive_list = []
    test_negative_list = []
    dataset_list = [train_postive_list, train_negative_list,
                    test_postive_list, test_negative_list]
    for index, dataset_dir in enumerate(dataset_dir_list):
        for file in pathlib.Path(os.path.join(base_url, dataset_dir)).iterdir():
            with open(os.path.join(os.path.join(base_url, dataset_dir),
                                file.name), 'r', encoding='utf-8') as file:
                dataset_list[index].append(file.read())
    print(len(train_postive_list), len(train_negative_list),
          len(test_postive_list), len(test_negative_list))
    print(train_postive_list[0])
