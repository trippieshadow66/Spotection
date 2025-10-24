from src.db import init_db, save_detection_result, get_latest_detection


init_db()
save_detection_result("data/frames/test.jpg", "overlays/test_overlay.jpg",
                      occupied_count=3, free_count=7,
                      stall_status={"1": True, "2": False, "3": True})
print(get_latest_detection())

