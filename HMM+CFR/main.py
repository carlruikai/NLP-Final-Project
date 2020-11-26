from data import build_corpus
from evaluate import hmm_train_eval, crf_train_eval, ensemble_evaluate


def main():
    """训练模型，评估结果"""

    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # 训练评估ｈｍｍ模型
    print("正在训练评估HMM模型...")
    hmm_pred = hmm_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id,
        tag2id
    )

    # 训练评估CRF模型
    print("正在训练评估CRF模型...")
    crf_pred = crf_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists)
    )


    ensemble_evaluate(
        [hmm_pred, crf_pred],
        test_tag_lists
    )


if __name__ == "__main__":
    main()
