import json, argparse, pathlib

# Please fill in your team information here
method = "idefics2"  # <str> -- name of the method
team = "Thinking Fast and Furious"  # <str> -- name of the team, !!!identical to the Google Form!!!
authors = ["Mete Ismayilzada", "Arina Rak", "Chun-tzu Chang"]  # <list> -- list of str, authors
email = "mismayilza@gmail.com"  # <str> -- e-mail address
institution = "EPFL"  # <str> -- institution or company
country = "Switzerland"  # <str> -- country or region


def main(input_path, output_path):
    with open(input_path, 'r') as file:
        output_res = json.load(file)

    output_res = [{"id": sample["id"], "question": sample["question_text"], "answer": sample["answer"]} for sample in output_res]

    submission_content = {
        "method": method,
        "team": team,
        "authors": authors,
        "email": email,
        "institution": institution,
        "country": country,
        "results": output_res
    }

    with open(output_path, 'w') as file:
        json.dump(submission_content, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=str, help="Path to submission inputs")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_path)
    output_path = input_path.with_name(f"{input_path.stem}_submission.json")
    main(input_path, output_path)
