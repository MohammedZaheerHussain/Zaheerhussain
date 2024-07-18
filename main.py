import hashlib


def crackHash(inputPass):
    try:
        passfile = open("wordlist.txt","r")
    except:
        print("Could not find the file.")

    for password in passfile:
        encPass = password.encode("utf-8")
        digest = hashlib.md5(encPass.strip()).hexdigest()
        if digest == inputPass:
            print("Password Found:" + password)

if __name__ == '__main__':
    crackHash("5f4dcc3b5aa765d61d8327deb882cf99")