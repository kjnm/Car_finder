import os

def main():
    with open(r'D:\to_move.txt') as f:
        lines = f.readlines()


    for line in lines:
        print(line[21:len(line)-2])

        try:

            if line[21:25] == "part":
                old = r'D:\imagesType\part' +  line[25:len(line)-2]
                new = r'D:\imagesType\full' +  line[25:len(line)-2]
                os.replace(old, new)
            else:
                old = r'D:\imagesType\full' +  line[25:len(line)-2]
                new = r'D:\imagesType\part' +  line[25:len(line)-2]
                os.replace(old, new)
        except FileNotFoundError:
            print("Not exist: ", line)

if __name__ == '__main__':
    main()