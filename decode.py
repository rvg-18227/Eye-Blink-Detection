alpha_to_morse = {'a': ".-", 'b': "-...", 'c': "-.-.", 'd': "-..", 'e': ".", 'f': "..-.", 'g': "--.", 'h': "....",
                  'i': "..", 'j': ".---", 'k': "-.-", 'l': ".-..", 'm': "--", 'n': "-.", 'o': "---", 'p': ".--.",
                  'q': "--.-", 'r': ".-.", 's': "...", 't': "-", 'u': "..-", 'v': "...-", 'w': ".--", 'x': "-..-",
                  'y': "-.--", 'z': "--..",
                  '1': ".----", '2': "..---", '3': "...--", '4': "....-", '5': ".....", '6': "-....", '7': "--...",
                  '8': "---..", '9': "----.", '0': "-----",
                  ' ': "¦", '.': ".-.-.-", ',': "--..--", '?': "..--..", "'": ".----.", '@': ".--.-.", '-': "-....-",
                  '"': ".-..-.", ':': "---...", ';': "---...", '=': "-...-", '!': "-.-.--", '/': "-..-.", '(': "-.--.",
                  ')': "-.--.-"}

morse_to_alpha = {value: key for key, value in alpha_to_morse.items()}


def from_morse(msg):
    result = ""
    for i in msg.split('/'):
        if i in morse_to_alpha:
            result += morse_to_alpha.get(i)
        else:
            if i != "":
                print(i, "could not be translated.")
    return result
