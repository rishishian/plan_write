import nltk
import numpy as np
import os
from torchtext.data import Field, TabularDataset


class TitleLine(TabularDataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.story_line)


class LineStory(TabularDataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.story_line)


VOCAB = Field(init_token='<sos>', eos_token='<eos>', lower=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def bleu_score(ref, can):
    ref = ref.strip().split()
    can = can.strip().split()
    # smooth = nltk.translate.bleu_score.SmoothingFunction()
    # return nltk.translate.bleu_score.sentence_bleu([ref], can, smoothing_function=smooth.method4)
    return nltk.translate.bleu_score.sentence_bleu([ref], can)


def test_bleu(result_path):
    GT_PATH = './data/test_story.txt'
    with open(GT_PATH, encoding="utf-8") as f:
        refs = f.readlines()
        refs = [''.join(l.split('</s>')) for l in refs]

    with open(result_path, encoding='utf-8') as f:
        raw_sents = f.readlines()
        cans = []
        for l in raw_sents:
            for sep in ['</s>', '<unk>', '<eos>']:
                l = l.replace(sep, '')
            cans.append(l.strip())

    assert len(cans) == len(refs), print(len(cans), len(refs))
    score_list = []
    for ref, can in zip(refs, cans):
        score_list.append(bleu_score(ref, can))
    sentence_bleu = np.mean(score_list)
    print(f'Sentence bleu score:{sentence_bleu}')


def create_testfile(generated_line_path='./result/title2line.txt'):
    # 这个函数设计的不太好，但是不想改了TAT
    # 此函数是想将生成的story line文件弄成拥有src,trg两个域的文件，这样就不用改loader的代码了
    base, ext = os.path.splitext(generated_line_path)
    ext = '_fortest.tsv'
    dest_file = base + ext
    print(f'Refining Test File(from:{generated_line_path}, to:{dest_file})')
    with open(generated_line_path, 'r') as f:
        lines = f.readlines()
        new_lines = [l.strip() + '\t' + f'STORY {i} TO BE GENERATED\n' for i, l in enumerate(lines)]
        with open(dest_file, 'w') as wf:
            for line in new_lines:
                wf.write(line)


def refine_result(raw_result_path):
    base, ext = os.path.splitext(raw_result_path)
    ext = ext.replace('.', '_refined.')
    refined_result_path = base + ext
    print(f'Refining Result File(from:{raw_result_path}, to:{refined_result_path})')
    refined_lines = []
    with open(raw_result_path, 'r') as f:
        raw_lines = f.readlines()
        for l in raw_lines:
            for sep in ['</s>', '<unk>', '<eos>']:
                l = l.replace(sep, '')
            refined_lines.append(l.strip())
    TESTSET_SIZE = 8159
    with open(refined_result_path, 'w') as f:
        for l in refined_lines[-TESTSET_SIZE:]:
            f.write(l+'\n')
    return refined_result_path


if __name__ == '__main__':
    create_testfile()

    # # cans_path = '../rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test'
    # cans_path = '/data1/lxx/course/nlp/NLP/language-model/tutorial/bx.txt'
    # # cans_path = '/data1/lxx/course/nlp/NLP/language-model/tutorial/generated_story_gt.txt'
    # # cans_path = '/data1/lxx/course/nlp/NLP/language-model/tutorial/generated_story.txt'
    # # cans_path = '/data1/lxx/course/nlp/NLP/language-model/tutorial/title_five.txt'

    # cans_path = './result/xs_bs.txt'   # 0.011146367604214563
    # cans_path = './result/line2story.txt'   # 0.008701637764563463
    # cans_path = './result/line2story0.5.txt'  # 0.0022856594964501976
    # cans_path = './result/generated_story.txt'  # 0.006871091977711924  # 这是一个较早期的结果，但后来我咋复现不出来了呢
    # # cans_path = './result/generated_story_gt.txt'  # 0.03258570753399208
    # # test_bleu(cans_path)
    # refined_path = refine_result(cans_path)
    # test_bleu(refined_path)
