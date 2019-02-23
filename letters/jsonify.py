import json
import os

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    
def main():
    LETTERS = {}
    subs = get_immediate_subdirectories(".")
    for sub in subs:
        LETTERS[sub] = os.listdir(f"./{sub}/")
    with open("letters.json","w+") as f:
        f.write(json.dumps(LETTERS))

if __name__=="__main__":
    main()