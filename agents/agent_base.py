from utils.event_helper import EventEmitter

class AgentBaseCommand(EventEmitter):
    def execute(self, args: any) -> None:
        raise NotImplementedError("Method not implemented.")
