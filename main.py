import viewwizard.session as vsession


def main():
    menu_context: vsession.MenuContext = vsession.MenuContext(vsession.ProgramSession())
    is_running: bool = True

    while is_running:
        is_running = vsession.prompt(menu_context)


if __name__ == "__main__":
    main()
