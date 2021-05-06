from models.transweigh import Transweigh
from models.transweigh_joint import TransweighJoint
from models.full_additive import FullAdditive
from models.classification_models.transweigh_twoword_classifier import TransweighTwoWordClassifier
from models.classification_models.basic_twoword_classifier import BasicTwoWordClassifier
from models.classification_models.single_word_classifier import SingleWordClassifier
from models.classification_models.joint_twoword_classifier import JointTwoWordClassifier
from models.classification_models.single_word_bert_classifier import SingleWordBertClassifier
from models.classification_models.layerweighting_twoword_classifier import BasicBertTwoWordClassifier
from data.dataloader import RankingDataset, JointRankingDataset, ClassificationDataset, extract_all_labels, \
    create_label_encoder, ContextualizedPhraseDataset, JointClassificationDataset, ContextualizedSemPhraseDataset, \
    ContextualizedRankingDataset


def init_classifier(config):
    """
    This method initialized the classifier with parameter specified in the config file
    :param config: the configuration
    :return: a torch classifier
    """
    classifier = None
    if config["model"]["type"] == "tw_single":
        classifier = Transweigh(input_dim=config["model"]["input_dim"],
                                dropout_rate=config["model"]["dropout"],
                                transformations=config["model"]["transformations"],
                                normalize_embeddings=config["model"]["normalize_embeddings"])
    if config["model"]["type"] == "tw_joint":
        classifier = TransweighJoint(input_dim=config["model"]["input_dim"],
                                     dropout_rate=config["model"]["dropout"],
                                     transformations=config["model"]["transformations"],
                                     normalize_embeddings=config["model"]["normalize_embeddings"])
    if config["model"]["type"] == "full_additive":
        classifier = FullAdditive(input_dim=config["model"]["input_dim"],
                                  normalize_embeddings=config["model"]["normalize_embeddings"])
    if config["model"]["type"] == "tw_classifier":
        classifier = TransweighTwoWordClassifier(input_dim=config["model"]["input_dim"],
                                                 dropout_rate=config["model"]["dropout"],
                                                 transformations=config["model"]["transformations"],
                                                 normalize_embeddings=config["model"]["normalize_embeddings"],
                                                 hidden_dim=config["model"]["hidden_dim"],
                                                 label_nr=config["model"]["labelnr"], add_single_words=True)
    if config["model"]["type"] == "basic_classifier":
        classifier = SingleWordBertClassifier(input_dim=config["model"]["input_dim"],
                                                layer_num=config["model"]["layers"],
                                            dropout_rate=config["model"]["dropout"],
                                            hidden_dim=config["model"]["hidden_dim"],
                                            label_nr=config["model"]["labelnr"]
                                            )
    if config["model"]["type"] == "static_classifier":
        classifier = SingleWordClassifier(input_dim=config["model"]["input_dim"],
                                            dropout_rate=config["model"]["dropout"],
                                            hidden_dim=config["model"]["hidden_dim"],
                                            label_nr=config["model"]["labelnr"]
                                             )
    if config["model"]["type"] == "basic_semclass_classifier":
        classifier = JointTwoWordClassifier(input_dim=config["model"]["input_dim"],
                                            dropout_rate=config["model"]["dropout"],
                                            hidden_dim=config["model"]["hidden_dim"],
                                            label_nr=config["model"]["labelnr"]
                                            )

    assert classifier, "no valid classifier name specified in the configuration"
    return classifier


def get_datasets(config):
    """
    Returns the datasets with the corresponding features (defined in the config file)
    :param config: the configuration file
    :return: training, validation, test dataset
    """
    mod = config["data_loader"]["modifier"]
    head = config["data_loader"]["head"]
    if config["model"]["type"] == "tw_joint":
        label_1 = config["data_loader"]["label_1"]
        label_2 = config["data_loader"]["label_2"]
        dataset_train = JointRankingDataset(data_path=config["train_data_path"],
                                            general_embedding_path=config["feature_extractor"]["general_embeddings"],
                                            label_embedding_path=config["feature_extractor"]["label_embeddings"],
                                            separator=config["data_loader"]["sep"],
                                            label_1=label_1, label_2=label_2, mod=mod, head=head)
        dataset_valid = JointRankingDataset(data_path=config["validation_data_path"],
                                            general_embedding_path=config["feature_extractor"]["general_embeddings"],
                                            label_embedding_path=config["feature_extractor"]["label_embeddings"],
                                            separator=config["data_loader"]["sep"],
                                            label_1=label_1, label_2=label_2, mod=mod, head=head)
        dataset_test = JointRankingDataset(data_path=config["test_data_path"],
                                           general_embedding_path=config["feature_extractor"]["general_embeddings"],
                                           label_embedding_path=config["feature_extractor"]["label_embeddings"],
                                           separator=config["data_loader"]["sep"],
                                           label_1=label_1, label_2=label_2, mod=mod, head=head)
    elif "classifier" in config["model"]["type"]:
        if config["feature_extractor"]["contextualized_embeddings"] is True:
            bert_parameter = config["feature_extractor"]["contextualized"]["bert"]
            bert_model = bert_parameter["model"]
            max_len = bert_parameter["max_sent_len"]
            lower_case = bert_parameter["lower_case"]
            batch_size = bert_parameter["batch_size"]
            label = config["data_loader"]["label"]
            load_bert = config["data_loader"]["load_bert_embeddings"]
            all_labels = extract_all_labels(training_data=config["train_data_path"],
                                            validation_data=config["validation_data_path"],
                                            test_data=config["test_data_path"], separator=config["data_loader"]["sep"],
                                            label=label
                                            )
            label_encoder = create_label_encoder(all_labels)
            print("labelsize %d" % len(set(all_labels)))
            if "semclass" in config["model"]["type"]:
                semclass = config["data_loader"]["semclass"]
                dataset_train = ContextualizedSemPhraseDataset(data_path=config["train_data_path"],
                                                               bert_model=bert_model, lower_case=lower_case,
                                                               max_len=max_len, separator=config["data_loader"]["sep"],
                                                               batch_size=batch_size, label_encoder=label_encoder,
                                                               label=label, mod=mod, head=head, low_layer=0,
                                                               top_layer=4,
                                                               load_bert_embeddings=load_bert, semclass=semclass)
                dataset_valid = ContextualizedSemPhraseDataset(data_path=config["validation_data_path"],
                                                               bert_model=bert_model, lower_case=lower_case,
                                                               max_len=max_len, separator=config["data_loader"]["sep"],
                                                               batch_size=batch_size, label_encoder=label_encoder,
                                                               label=label, mod=mod, head=head, low_layer=0,
                                                               top_layer=4,
                                                               load_bert_embeddings=load_bert, semclass=semclass)
                dataset_test = ContextualizedSemPhraseDataset(data_path=config["test_data_path"],
                                                              bert_model=bert_model, lower_case=lower_case,
                                                              max_len=max_len, separator=config["data_loader"]["sep"],
                                                              batch_size=batch_size, label_encoder=label_encoder,
                                                              label=label, mod=mod, head=head, low_layer=0, top_layer=4,
                                                              load_bert_embeddings=load_bert, semclass=semclass)
            else:
                dataset_train = ContextualizedPhraseDataset(data_path=config["train_data_path"],
                                                            bert_model=bert_model, lower_case=lower_case,
                                                            max_len=max_len, separator=config["data_loader"]["sep"],
                                                            batch_size=batch_size, label_encoder=label_encoder,
                                                            label=label, mod=mod, head=head, low_layer=0, top_layer=4,
                                                            load_bert_embeddings=load_bert)
                dataset_valid = ContextualizedPhraseDataset(data_path=config["validation_data_path"],
                                                            bert_model=bert_model, lower_case=lower_case,
                                                            max_len=max_len, separator=config["data_loader"]["sep"],
                                                            batch_size=batch_size, label_encoder=label_encoder,
                                                            label=label, mod=mod, head=head, low_layer=0, top_layer=4,
                                                            load_bert_embeddings=load_bert)
                dataset_test = ContextualizedPhraseDataset(data_path=config["test_data_path"],
                                                           bert_model=bert_model, lower_case=lower_case,
                                                           max_len=max_len, separator=config["data_loader"]["sep"],
                                                           batch_size=batch_size, label_encoder=label_encoder,
                                                           label=label, mod=mod, head=head, low_layer=0, top_layer=4,
                                                           load_bert_embeddings=load_bert)

        else:

            label = config["data_loader"]["label"]
            all_labels = extract_all_labels(training_data=config["train_data_path"],
                                            validation_data=config["validation_data_path"],
                                            test_data=config["test_data_path"], separator=config["data_loader"]["sep"],
                                            label=config["data_loader"]["label"]
                                            )
            print("all labels")
            print(all_labels)
            label_encoder = create_label_encoder(all_labels)
            print("labelsize %d" % len(set(all_labels)))
            if "semclass" in config["model"]["type"]:
                semclass = config["data_loader"]["semclass"]
                dataset_train = JointClassificationDataset(data_path=config["train_data_path"],
                                                           embedding_path=config["feature_extractor"][
                                                               "general_embeddings"],
                                                           separator=config["data_loader"]["sep"],
                                                           label=label, mod=mod, head=head, label_encoder=label_encoder,
                                                           feature=semclass)
                dataset_valid = JointClassificationDataset(data_path=config["validation_data_path"],
                                                           embedding_path=config["feature_extractor"][
                                                               "general_embeddings"],
                                                           separator=config["data_loader"]["sep"],
                                                           label=label, mod=mod, head=head, label_encoder=label_encoder,
                                                           feature=semclass)
                dataset_test = JointClassificationDataset(data_path=config["test_data_path"],
                                                          embedding_path=config["feature_extractor"][
                                                              "general_embeddings"],
                                                          separator=config["data_loader"]["sep"],
                                                          label=label, mod=mod, head=head, label_encoder=label_encoder,
                                                          feature=semclass)
            else:

                dataset_train = ClassificationDataset(data_path=config["train_data_path"],
                                                      embedding_path=config["feature_extractor"]["general_embeddings"],
                                                      separator=config["data_loader"]["sep"],
                                                      label=label, mod=mod, head=head, label_encoder=label_encoder)
                dataset_valid = ClassificationDataset(data_path=config["validation_data_path"],
                                                      embedding_path=config["feature_extractor"]["general_embeddings"],
                                                      separator=config["data_loader"]["sep"],
                                                      label=label, mod=mod, head=head, label_encoder=label_encoder)
                dataset_test = ClassificationDataset(data_path=config["test_data_path"],
                                                     embedding_path=config["feature_extractor"]["general_embeddings"],
                                                     separator=config["data_loader"]["sep"],
                                                     label=label, mod=mod, head=head, label_encoder=label_encoder)

    else:
        label = config["data_loader"]["label"]
        if config["feature_extractor"]["contextualized_embeddings"] is True:
            bert_parameter = config["feature_extractor"]["contextualized"]["bert"]
            bert_model = bert_parameter["model"]
            max_len = bert_parameter["max_sent_len"]
            lower_case = bert_parameter["lower_case"]
            batch_size = bert_parameter["batch_size"]
            load_bert = config["data_loader"]["load_bert_embeddings"]
            load_labels = config["data_loader"]["load_labels"]
            label_definition_path = config["feature_extractor"]["definition"]
            dataset_train = ContextualizedRankingDataset(data_path=config["train_data_path"],
                                                         bert_model=bert_model, lower_case=lower_case,
                                                         max_len=max_len, separator=config["data_loader"]["sep"],
                                                         batch_size=batch_size,
                                                         label_definition_path=label_definition_path,
                                                         label=label, mod=mod, head=head, low_layer=0, top_layer=4,
                                                         load_bert_embeddings=load_bert,
                                                         load_label_embeddings=load_labels)
            dataset_valid = ContextualizedRankingDataset(data_path=config["validation_data_path"],
                                                         bert_model=bert_model, lower_case=lower_case,
                                                         max_len=max_len, separator=config["data_loader"]["sep"],
                                                         batch_size=batch_size,
                                                         label=label, mod=mod, head=head, low_layer=0, top_layer=4,
                                                         load_bert_embeddings=load_bert,
                                                         label_definition_path=label_definition_path,
                                                         load_label_embeddings=load_labels)
            dataset_test = ContextualizedRankingDataset(data_path=config["test_data_path"],
                                                        bert_model=bert_model, lower_case=lower_case,
                                                        max_len=max_len, separator=config["data_loader"]["sep"],
                                                        batch_size=batch_size,
                                                        label=label, mod=mod, head=head, low_layer=0, top_layer=4,
                                                        load_bert_embeddings=load_bert,
                                                        label_definition_path=label_definition_path,
                                                        load_label_embeddings=load_labels)
        else:
            dataset_train = RankingDataset(data_path=config["train_data_path"],
                                           general_embedding_path=config["feature_extractor"]["general_embeddings"],
                                           label_embedding_path=config["feature_extractor"]["label_embeddings"],
                                           separator=config["data_loader"]["sep"],
                                           label=label, mod=mod, head=head)
            dataset_valid = RankingDataset(data_path=config["validation_data_path"],
                                           general_embedding_path=config["feature_extractor"]["general_embeddings"],
                                           label_embedding_path=config["feature_extractor"]["label_embeddings"],
                                           separator=config["data_loader"]["sep"],
                                           label=label, mod=mod, head=head)
            dataset_test = RankingDataset(data_path=config["test_data_path"],
                                          general_embedding_path=config["feature_extractor"]["general_embeddings"],
                                          label_embedding_path=config["feature_extractor"]["label_embeddings"],
                                          separator=config["data_loader"]["sep"],
                                          label=label, mod=mod, head=head)

    return dataset_train, dataset_valid, dataset_test
