def extract_answer(solution_text: str):
    """
    Extract the final answer from a solution string by taking the content of the last \boxed{...}.

    Returns:
        - str: extracted answer (may be empty string if \boxed{} is present with no content)
        - None: if no boxed answer is found
    """

    answer = remove_boxed(last_boxed_only_string(solution_text))
    answer = answer.replace("**", "")

    if not answer:
        if isinstance(answer, str) and len(answer) == 0:
            # Is empty string, check if '\\boxed{}' is present (if present, extracted answer is empty string)
            if "\\boxed{}" in solution_text:
                return ""
        return None

    return answer

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return ""

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = ""
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return ""