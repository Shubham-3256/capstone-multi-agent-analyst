"""Unit tests for the thread-safe MemoryStore caching layer."""

import time
import threading
import pytest
from src.memory.memory_store import MemoryStore


def test_memory_store_basic_operations():
    """Test core set, get, exists, delete, and clear operations inside MemoryStore."""
    store = MemoryStore()
    
    # 1. Set & Get
    store.set("user", "antigravity")
    assert store.get("user") == "antigravity"
    assert store.exists("user") is True
    
    # 2. Update value
    store.set("user", "agent")
    assert store.get("user") == "agent"
    
    # 3. Delete
    deleted = store.delete("user")
    assert deleted is True
    assert store.get("user") is None
    assert store.exists("user") is False
    
    # 4. Clear
    store.set("a", 1)
    store.set("b", 2)
    assert store.get_size() == 2
    store.clear()
    assert store.get_size() == 0


def test_memory_store_ttl_expiration():
    """Test cache Time-To-Live (TTL) key evictions."""
    store = MemoryStore()
    
    # Set a key with 1-second TTL expiration
    store.set("temp_key", "expires", ttl_seconds=1)
    assert store.get("temp_key") == "expires"
    assert store.exists("temp_key") is True
    
    # Wait for key to expire
    time.sleep(1.2)
    
    # Verify key is evicted
    assert store.get("temp_key") is None
    assert store.exists("temp_key") is False
    assert store.get_size() == 0


def test_memory_store_thread_safety():
    """Verify thread-safety of MemoryStore when accessed by parallel writing threads."""
    store = MemoryStore()
    num_threads = 10
    loops = 100
    
    def worker(thread_idx):
        for i in range(loops):
            key = f"key_{thread_idx}_{i}"
            store.set(key, i)
            # Fetch to simulate read/write operations
            store.get(key)
            
    threads = []
    for idx in range(num_threads):
        t = threading.Thread(target=worker, args=(idx,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    # Verify size
    assert store.get_size() == num_threads * loops
