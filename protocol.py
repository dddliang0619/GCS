def encode_result(result):
    if not result.valid:
        return "$VIS,0,NONE,0,0,0,0*\n"

    return "$VIS,1,{},{},{},{},{}*\n".format(
        result.color,
        result.dx,
        result.dy,
        result.pixels,
        result.confidence,
    )


def parse_command(line):
    if not line:
        return None, None

    try:
        text = line.decode().strip() if isinstance(line, bytes) else line.strip()
        if not (text.startswith("$CMD,") and text.endswith("*")):
            return None, None

        fields = text[1:-1].split(",")
        command = fields[1]
        argument = fields[2] if len(fields) > 2 else None
        return command, argument
    except Exception:
        return None, None

