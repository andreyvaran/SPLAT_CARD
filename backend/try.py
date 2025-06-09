import asyncio
import random
import aiohttp
import time


async def stress_test(url, n_requests, loop):
    """Асинхронная функция для стресс-тестирования HTTP-запроса."""
    start_time = time.time()
    tasks = []
    for _ in range(n_requests):
        url = f"http://0.0.0.0:8080/api/author/{random.randint(1, 1000)}"
        tasks.append(loop.create_task(fetch_request(url)))

    responses = await asyncio.gather(*tasks)
    failed_responses = [response for response in responses if response.status == 500]
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"{len(failed_responses)} requests failed out of {n_requests}, which is {len(failed_responses)/n_requests*100}%."
    )
    print(f"Total time taken: {elapsed_time}")


async def fetch_request(url):
    """Асинхронная функция для отправки HTTP-запроса."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return response


if __name__ == "__main__":
    url = ""  # Замените на реальный URL
    n_requests = 5  # Количество запросов для тестирования
    loop = asyncio.get_event_loop()
    loop.run_until_complete(stress_test(url, n_requests, loop))
    loop.close()
