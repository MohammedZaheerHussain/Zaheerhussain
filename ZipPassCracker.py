import zipfile

def extractFile(zfile, password):
    try:
        zfile.extractall(pwd=bytes(password, 'utf-8'))
        return password
    except:
        print('wrong pass')
        return

def main():
    zfile = zipfile.ZipFile('test.zip')
    passFile = open('passlist.txt')
    for line in passFile.readlines():
        password = line.strip('\n')
        guess = extractFile(zfile, password)
        if guess:
            print('Password =' + password)
            break


if __name__ == '__main__':
    main()