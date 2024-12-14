import json
from src.linguistic_utils import get_normal_form


def read_and_process_synonym_dict(path_to_dict):
    with open(path_to_dict, 'r', encoding='utf-8') as file:
        dict_ = json.load(file)

    dict_processed = {}
    for i in dict_:
        if len(i["synsets"]) == 0:
            continue

        if i["lemma"].lower() in dict_processed:
            for synset in i["synsets"]:
                dict_processed[i["lemma"].lower()].extend(synset["clean"])
        else:
            dict_processed[i["lemma"].lower()] = []

            for synset in i["synsets"]:
                dict_processed[i["lemma"].lower()].extend(synset["clean"])

        dict_processed[i["lemma"].lower()] = list(set(dict_processed[i["lemma"].lower()]))
    return dict_processed


def remove_bad_parsed_synonyms(synonym_dict):

    remove_list = ["пестл", "розм", "від", "за", "жм", "як зв", "жарт", "тйж-ба"]

    for key in synonym_dict:
        synonym_dict[key] = [word for word in synonym_dict[key] if word not in remove_list]

    return synonym_dict


def remove_bad_synonyms(synonym_dict):
    bad_synonyms_to_remove = [
        ("чоловік", "жінка"),
        ("жінка", "чоловік"),
        ("місце", "ід"),
        ("стояти", "клячати"),
        ("грати", "зображати"),
        ("люди", "мир"),
        ("смугастий", "із смугами"),
        ("сонце", "приязнь"),
        ("вода", "курорт"),
        ("чоловік", "подружжя"),
        ("чоловік", "дружина"),
        ("хлопець", "дівчина"),
        ("білий", "чорний"),
        ("рука", "у десна"),
        ("сидіти", "проживати"),
        ("сидіти", "(на"),
        ("молодий", "старий")
    ]
    for word, bad_synonym in bad_synonyms_to_remove:
        if word in synonym_dict and bad_synonym in synonym_dict[word]:
            synonym_dict[word].remove(bad_synonym)


def get_all_synonyms(word, synonym_dict):
    normal_form = get_normal_form(word)

    if word in synonym_dict:
        return synonym_dict[word]
    elif normal_form in synonym_dict:
        return synonym_dict[normal_form]
    else:
        return []


def prepare_synonym_dict(path_to_synonym_dict):
    synonym_dict = read_and_process_synonym_dict(path_to_synonym_dict)
    synonym_dict = remove_bad_parsed_synonyms(synonym_dict)
    remove_bad_synonyms(synonym_dict)
    return synonym_dict
