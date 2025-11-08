import asyncio


class Topic:
    """Simple async topic with multiple subscribers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        """Create and return a new subscription queue."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._subscribers.append(queue)
        return queue

    async def publish(self, message: object) -> None:
        """Publish message to all subscribers (drop if queue full)."""
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(message)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass


class PubSub:
    """Pub/sub system."""

    def __init__(self) -> None:
        self.frame = Topic("frame")
        self.target_location = Topic("target_location")
        self.servo_state = Topic("servo_state")
        self.pose_data = Topic("pose_data")
