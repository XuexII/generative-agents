import ahocorasick
A = ahocorasick.Automaton()


print()
for index,word in enumerate("he her hers she".split()):
    A.add_word(word, (index, word))
