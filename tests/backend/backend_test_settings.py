"""Settings shared by the backend tests."""

# Bootstrap samples per submission in tests. Production uses config.BOOTSTRAP_SAMPLES
# (10,000); the stored bootstrap fixtures in test_data/ have this many (see
# generate_test_data.py), and scoring in tests is 10x faster.
TEST_BOOTSTRAP_SAMPLES = 1000
