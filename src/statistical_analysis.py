import argparse

def run(mode_name: str):
    print(f'Analyzing model: {model_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical analysis: name model to analyze .')
    parser.add_argument('model_name', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    model_name = args.model_name

    run(model_name)