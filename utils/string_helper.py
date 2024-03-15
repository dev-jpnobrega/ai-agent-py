def interpolate(input_string: str, args: dict, options: dict = {"openedWith": "(", "closedWith": ")"}) -> str:
    interpolated_string = input_string

    for key, value in args.items():
        interpolated_string = interpolated_string.replace(f"{options['openedWith']}{key}{options['closedWith']}", str(value))

    return interpolated_string
