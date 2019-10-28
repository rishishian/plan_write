from tqdm import tqdm

# split src data to (title+line) and (line+story)
src_data_path_list = ['train_title_line_story.txt', 'valid_title_line_story.txt', 'test_title_line_story.txt']
t2l_data_path_list = ['train_title_line.tsv', 'valid_title_line.tsv', 'test_title_line.tsv']
l2s_data_path_list = ['train_line_story.tsv', 'valid_line_story.tsv', 'test_line_story.tsv']


def parse_line(line):
    title, rest = line.split('<EOT>')
    story_line, story = rest.split('<EOL>')
    return title.strip(), story_line.strip(), story.strip()


for src_data_path, t2l_data_path, l2s_data_path in zip(src_data_path_list, t2l_data_path_list, l2s_data_path_list):
    with open(src_data_path, 'r') as src_file:
        with open(t2l_data_path, 'w') as t2l_file:
            with open(l2s_data_path, 'w') as l2s_file:
                print(f'Processing {src_data_path}')
                src_lines = src_file.readlines()
                for line in tqdm(src_lines):
                    title, story_line, story = parse_line(line)
                    t2l_file.write(title + '\t' + story_line + '\n')
                    l2s_file.write(story_line + '\t' + story + '\n')

# ground-truth story for testset
gt_testset_path = 'test_story.txt'
with open(src_data_path_list[-1], 'r') as src_file:
    with open(gt_testset_path, 'w') as gt_file:
        src_lines = src_file.readlines()
        for line in tqdm(src_lines):
            title, story_line, story = parse_line(line)
            gt_file.write(story + '\n')

