import tarfile

parts = ['splits/shanghaitech.tar.gz.aa', 'splits/shanghaitech.tar.gz.ab', 'splits/shanghaitech.tar.gz.ac', 'splits/shanghaitech.tar.gz.ad', 'splits/shanghaitech.tar.gz.ae', 'splits/shanghaitech.tar.gz.af', 'splits/shanghaitech.tar.gz.ag']
with open('TarExtract/combined.tar', 'wb') as outfile:
    for fname in parts:
        with open(fname, 'rb') as part:
            outfile.write(part.read())