from src.resources import morph


def get_normal_form(word):
    parsed = morph.parse(word)
    if parsed:
        return parsed[0].normal_form
    else:
        return word
