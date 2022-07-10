from spellchecker import SpellChecker

spell= SpellChecker()

text= """
Personne n'est plus généreux et énergique que toi ! Ta devise ?


✨ "Hey ! Ecoutez mon histoire !" ✨
"""
misspelled= spell.unknown(text)

for word in misspelled:
    print(word, spell.correction(word))
