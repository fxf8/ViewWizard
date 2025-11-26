import viewwizard.session as vsession


def main():
    menu_context: vsession.MenuContext = vsession.MenuContext()

    while True:
        vsession.prompt(menu_context)


if __name__ == "__main__":
    main()
