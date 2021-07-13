import phonetic_alphabet as alpha

letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def phonetic_from_int(i: int) -> str:
    if i >= len(letters):
        raise ValueError('Phonetic index too high')
    return alpha.read(letters[i]).title()
