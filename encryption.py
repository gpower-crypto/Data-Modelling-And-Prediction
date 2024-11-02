import streamlit as st

def caesar_cipher_encrypt(plaintext, shift):
    encrypted_text = ""
    for char in plaintext:
        if char.isalpha():
            if char.islower():
                encrypted_text += chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            else:
                encrypted_text += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
        else:
            encrypted_text += char
    return encrypted_text

def caesar_cipher_decrypt(ciphertext, shift):
    decrypted_text = ""
    for char in ciphertext:
        if char.isalpha():
            if char.islower():
                decrypted_text += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
            else:
                decrypted_text += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
        else:
            decrypted_text += char
    return decrypted_text

def is_valid_english(text):
    common_english_words = set([
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"
    ])
    words = text.split()
    valid_word_count = sum(word.lower() in common_english_words for word in words)
    return valid_word_count > 0  # If there's at least one common English word, we consider it valid

def brute_force_attack(ciphertext):
    possible_messages = []
    for shift in range(1, 26):
        possible_message = caesar_cipher_decrypt(ciphertext, shift)
        if is_valid_english(possible_message):
            possible_messages.append((shift, possible_message))
    return possible_messages

def main():
    st.title("Caesar Cipher Encryption, Decryption, and Brute Force Attack")

    # Select mode: Encryption, Decryption, or Brute Force Attack
    mode = st.radio("Select mode:", ("Encryption", "Decryption", "Brute Force Attack"))

    if mode == "Encryption":
        plaintext = st.text_input("Enter plaintext message:")
        key = st.slider("Select shift key:", 1, 25, 3)
        if st.button("Encrypt"):
            ciphertext = caesar_cipher_encrypt(plaintext, key)
            st.success(f"Ciphertext: {ciphertext}")

    elif mode == "Decryption":
        ciphertext = st.text_input("Enter ciphertext message:")
        key = st.slider("Select shift key:", 1, 25, 3)
        if st.button("Decrypt"):
            decrypted_text = caesar_cipher_decrypt(ciphertext, key)
            st.success(f"Decrypted text: {decrypted_text}")

    elif mode == "Brute Force Attack":
        ciphertext = st.text_input("Enter ciphertext message:")
        if st.button("Brute Force Attack"):
            possible_messages = brute_force_attack(ciphertext)
            if possible_messages:
                st.success("Possible decrypted messages with their keys:")
                for shift, message in possible_messages:
                    st.write(f"Key: {shift}, Message: {message}")
            else:
                st.error("No valid English text found.")

if __name__ == "__main__":
    main()
