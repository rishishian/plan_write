import os


class Config:
    DataRootPath = './data/'
    ModelRootPath = './model'
    ResultRootPath = './result'
    Title2Line = {
        # data
        'trainset': os.path.join(DataRootPath, 'train_title_line.tsv'),
        'validset': os.path.join(DataRootPath, 'valid_title_line.tsv'),
        'testset': os.path.join(DataRootPath, 'test_title_line.tsv'),
        'src_field': 'title',
        'trg_field': 'story_line',
        'min_freq': 25,

        # model
        'enc_emb_dim': 256,
        'dec_emb_dim': 256,
        'enc_hid_dim': 256,
        'dec_hid_dim': 256,
        'n_layers': 1,
        'enc_dropout': 0,
        'dec_dropout': 0,
        'model_path': os.path.join(ModelRootPath, 'title2line.pt'),
        'result_path': os.path.join(ResultRootPath, 'title2line.txt'),
        'teacher_force': 1.0,

        # train
        'batch_size': 128,
        'n_epoch': 10,
        'clip_norm': 1,
    }
    Line2Story = {
        # data
        'trainset': os.path.join(DataRootPath, 'train_line_story.tsv'),
        'validset': os.path.join(DataRootPath, 'valid_line_story.tsv'),
        # 'testset': os.path.join(DataRootPath, 'test_line_story.tsv'),
        'testset': os.path.join(DataRootPath, 'test_gnr_line_STORY.tsv'),
        'src_field': 'story_line',
        'trg_field': 'story',
        'min_freq': 25,

        # model
        'enc_emb_dim': 256,
        'dec_emb_dim': 256,
        'enc_hid_dim': 256,
        'dec_hid_dim': 256,
        'n_layers': 1,
        'enc_dropout': 0,
        'dec_dropout': 0,
        'model_path': os.path.join(ModelRootPath, 'line2story.pt'),
        'result_path': os.path.join(ResultRootPath, 'line2story_wo_repeat.txt'),
        'teacher_force': 1.0,

        # train
        'batch_size': 128,
        'n_epoch': 10,
        'clip_norm': 1,
    }
    Title2LineTest = {
        # data
        'trainset': os.path.join(DataRootPath, 'train_title_line.tsv'),
        'validset': os.path.join(DataRootPath, 'valid_title_line.tsv'),
        'testset': os.path.join(DataRootPath, 'test_title_line.tsv'),
        'src_field': 'title',
        'trg_field': 'story_line',
        'min_freq': 25,

        # model
        'enc_emb_dim': 256,
        'dec_emb_dim': 256,
        'enc_hid_dim': 256,
        'dec_hid_dim': 256,
        'n_layers': 1,
        'enc_dropout': 0,
        'dec_dropout': 0,
        'model_path': os.path.join(ModelRootPath, 'title2line0.5.pt'),
        'result_path': os.path.join(ResultRootPath, 'title2line0.5.txt'),
        'teacher_force': 1.0,

        # train
        'batch_size': 128,
        'n_epoch': 10,
        'clip_norm': 1,
    }
    Line2StoryTest = {
        # data
        'trainset': os.path.join(DataRootPath, 'train_line_story.tsv'),
        'validset': os.path.join(DataRootPath, 'valid_line_story.tsv'),
        # 'testset': os.path.join(DataRootPath, 'test_line_story.tsv'),
        'testset': os.path.join(DataRootPath, 'test_gnr_line_STORY.tsv'),
        'src_field': 'story_line',
        'trg_field': 'story',
        'min_freq': 25,

        # model
        'enc_emb_dim': 256,
        'dec_emb_dim': 256,
        'enc_hid_dim': 256,
        'dec_hid_dim': 256,
        'n_layers': 1,
        'enc_dropout': 0,
        'dec_dropout': 0,
        'model_path': os.path.join(ModelRootPath, 'line2story0.5.pt'),
        'result_path': os.path.join(ResultRootPath, 'line2story0.5.txt'),
        'teacher_force': 0.5,

        # train
        'batch_size': 128,
        'n_epoch': 5,
        'clip_norm': 1,
    }
    Title2LineAttn = {
        # data
        'trainset': os.path.join(DataRootPath, 'train_title_line.tsv'),
        'validset': os.path.join(DataRootPath, 'valid_title_line.tsv'),
        'testset': os.path.join(DataRootPath, 'test_title_line.tsv'),
        'src_field': 'title',
        'trg_field': 'story_line',
        'min_freq': 25,

        # model
        'enc_emb_dim': 256,
        'dec_emb_dim': 256,
        'enc_hid_dim': 256,
        'dec_hid_dim': 256,
        'n_layers': 1,
        'enc_dropout': 0,
        'dec_dropout': 0,
        'model_path': os.path.join(ModelRootPath, 'title2line0.5_attn.pt'),
        'result_path': os.path.join(ResultRootPath, 'title2line0.5_attn.txt'),
        'teacher_force': 1.0,

        # train
        'batch_size': 128,
        'n_epoch': 10,
        'clip_norm': 1,
    }
    Line2StoryAttn = {
        # data
        'trainset': os.path.join(DataRootPath, 'train_line_story.tsv'),
        'validset': os.path.join(DataRootPath, 'valid_line_story.tsv'),
        # 'testset': os.path.join(DataRootPath, 'test_line_story.tsv'),
        # 'testset': os.path.join(DataRootPath, 'test_gnr_line_STORY.tsv'),
        'testset': './result/title2line_fortest.tsv',
        'src_field': 'story_line',
        'trg_field': 'story',
        'min_freq': 25,

        # model
        'enc_emb_dim': 256,
        'dec_emb_dim': 256,
        'enc_hid_dim': 256,
        'dec_hid_dim': 256,
        'n_layers': 1,
        'enc_dropout': 0,
        'dec_dropout': 0,
        'model_path': os.path.join(ModelRootPath, 'line2story0.5_attn.pt'),
        'result_path': os.path.join(ResultRootPath, 'line2story0.5_attn.txt'),
        'teacher_force': 0.5,

        # train
        'batch_size': 128,
        'n_epoch': 5,
        'clip_norm': 1,
    }


if __name__ == '__main__':
    config = getattr(Config, 'Title2Line')
    print(config['trainset'])
    print('test config over')
