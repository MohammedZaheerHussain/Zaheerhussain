from socket import *
def conScan(tgtHost, tgtPort):
    try:
        connskt = socket(AF_INET, SOCK_STREAM)
        connskt.connect((tgtHost,tgtPort))
        print('[+]%d/tcp open'%tgtPort)
        connskt.close()
    except:
        print('[-]%d/tcp closed'% tgtPort)

def portScan(tgtHost, tgtPort):
    try:
        tgtIP = gethostbyname(tgtHost)
    except:
        print('[-] Cannot resolve %s '% tgtHost)
        return
    try:
        tgtName = gethostbyaddr(tgtIP)
        print('\n [+] Scan Result of : %s ' % tgtName[0])
    except:
        print('\n [+] Scan result of: %s ' % tgtIP)
    setdefaulttimeout(1)
    for tgtPort in tgtPort:
        print('Scanning Port: %d' % tgtPort)
        conScan(tgtHost, int(tgtPort))

if __name__ == '__main__':
    portScan('google.com',[80, 22])