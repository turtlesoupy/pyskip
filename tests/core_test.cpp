#define CATCH_CONFIG_MAIN

#include <fmt/format.h>
#include <catch2/catch.hpp>

#include <skimpy/detail/core.hpp>

using namespace skimpy;

TEST_CASE("Test range store compression", "[range_compression]") {
  RangeStore<int, char> store(
      9, std::vector{0, 2, 4, 5, 7, 8}, {'a', 'b', 'b', 'c', 'c', 'd'});

  // Compress the ranges.
  store.compress();

  // Check that the compressed ranges make sense.
  auto gen = store.scan(0);
  REQUIRE(gen.next() == std::tuple(0, 2, 'a'));
  REQUIRE(gen.next() == std::tuple(2, 5, 'b'));
  REQUIRE(gen.next() == std::tuple(5, 8, 'c'));
  REQUIRE(gen.next() == std::tuple(8, 9, 'd'));
  REQUIRE(gen.done());
}

TEST_CASE("Test single range store scanning", "[single_range_scan]") {
  BufferedRangeStore<int, char> store(9, {0}, {'a'});

  {
    auto gen = store.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 9, 'a'));
    REQUIRE(gen.done());
  }

  {
    auto gen = store.scan(1);
    REQUIRE(gen.next() == std::tuple(0, 9, 'a'));
    REQUIRE(gen.done());
  }

  {
    auto gen = store.scan(8);
    REQUIRE(gen.next() == std::tuple(0, 9, 'a'));
    REQUIRE(gen.done());
  }

  REQUIRE_THROWS(store.scan(9));
}

TEST_CASE("Test multiple range store scanning", "[multi_range_scan]") {
  BufferedRangeStore<int, char> store(
      9, {0, 2, 4, 5, 7, 8}, {'a', 'b', 'c', 'd', 'e', 'f'});

  {
    auto gen = store.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 2, 'a'));
    REQUIRE(gen.next() == std::tuple(2, 4, 'b'));
    REQUIRE(gen.next() == std::tuple(4, 5, 'c'));
    REQUIRE(gen.next() == std::tuple(5, 7, 'd'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'e'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'f'));
    REQUIRE(gen.done());
  }

  {
    auto gen = store.scan(2);
    REQUIRE(gen.next() == std::tuple(2, 4, 'b'));
    REQUIRE(gen.next() == std::tuple(4, 5, 'c'));
    REQUIRE(gen.next() == std::tuple(5, 7, 'd'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'e'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'f'));
    REQUIRE(gen.done());
  }

  {
    auto gen = store.scan(6);
    REQUIRE(gen.next() == std::tuple(5, 7, 'd'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'e'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'f'));
    REQUIRE(gen.done());
  }

  {
    auto gen = store.scan(8);
    REQUIRE(gen.next() == std::tuple(8, 9, 'f'));
    REQUIRE(gen.done());
  }

  REQUIRE_THROWS(store.scan(9));
}

TEST_CASE("Test buffered range store scanning", "[buffered_range_scan]") {
  BufferedRangeStore<int, char> store(
      9, {0, 2, 4, 5, 8}, {'a', 'b', 'c', 'd', 'e'});

  store.buffer = {
      std::tuple(1, 3, '0'),
      std::tuple(4, 5, '1'),
      std::tuple(6, 7, '2'),
  };

  {
    auto gen = store.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 1, 'a'));
    REQUIRE(gen.next() == std::tuple(1, 3, '0'));
    REQUIRE(gen.next() == std::tuple(3, 4, 'b'));
    REQUIRE(gen.next() == std::tuple(4, 5, '1'));
    REQUIRE(gen.next() == std::tuple(5, 6, 'd'));
    REQUIRE(gen.next() == std::tuple(6, 7, '2'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'd'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'e'));
    REQUIRE(gen.done());
  }

  {
    auto gen = store.scan(1);
    REQUIRE(gen.next() == std::tuple(1, 3, '0'));
    REQUIRE(gen.next() == std::tuple(3, 4, 'b'));
    REQUIRE(gen.next() == std::tuple(4, 5, '1'));
    REQUIRE(gen.next() == std::tuple(5, 6, 'd'));
    REQUIRE(gen.next() == std::tuple(6, 7, '2'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'd'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'e'));
    REQUIRE(gen.done());
  }

  {
    auto gen = store.scan(2);
    REQUIRE(gen.next() == std::tuple(1, 3, '0'));
    REQUIRE(gen.next() == std::tuple(3, 4, 'b'));
    REQUIRE(gen.next() == std::tuple(4, 5, '1'));
    REQUIRE(gen.next() == std::tuple(5, 6, 'd'));
    REQUIRE(gen.next() == std::tuple(6, 7, '2'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'd'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'e'));
    REQUIRE(gen.done());
  }

  {
    auto gen = store.scan(3);
    REQUIRE(gen.next() == std::tuple(3, 4, 'b'));
    REQUIRE(gen.next() == std::tuple(4, 5, '1'));
    REQUIRE(gen.next() == std::tuple(5, 6, 'd'));
    REQUIRE(gen.next() == std::tuple(6, 7, '2'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'd'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'e'));
    REQUIRE(gen.done());
  }

  {
    auto gen = store.scan(5);
    REQUIRE(gen.next() == std::tuple(5, 6, 'd'));
    REQUIRE(gen.next() == std::tuple(6, 7, '2'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'd'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'e'));
    REQUIRE(gen.done());
  }

  {
    auto gen = store.scan(6);
    REQUIRE(gen.next() == std::tuple(6, 7, '2'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'd'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'e'));
    REQUIRE(gen.done());
  }

  {
    auto gen = store.scan(7);
    REQUIRE(gen.next() == std::tuple(7, 8, 'd'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'e'));
    REQUIRE(gen.done());
  }

  REQUIRE_THROWS(store.scan(9));
}

TEST_CASE("Test buffered range store flushing", "[buffered_range_flush]") {
  BufferedRangeStore<int, char> buffered_store(
      9, {0, 2, 4, 5, 7, 8}, {'a', 'b', 'c', 'd', 'e', 'f'});

  buffered_store.buffer = {
      std::tuple(1, 3, '0'),
      std::tuple(4, 5, '1'),
      std::tuple(5, 6, '2'),
  };

  buffered_store.flush();

  // Make sure that the buffer is empty.
  REQUIRE(buffered_store.buffer.empty());

  // Make sure that the ranges reflect what they should.
  {
    auto gen = buffered_store.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 1, 'a'));
    REQUIRE(gen.next() == std::tuple(1, 3, '0'));
    REQUIRE(gen.next() == std::tuple(3, 4, 'b'));
    REQUIRE(gen.next() == std::tuple(4, 5, '1'));
    REQUIRE(gen.next() == std::tuple(5, 6, '2'));
    REQUIRE(gen.next() == std::tuple(6, 7, 'd'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'e'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'f'));
    REQUIRE(gen.done());
  }

  // While we're at it let's compress things.
  buffered_store.store.compress();
  REQUIRE(buffered_store.store.starts.size() == 8);
  REQUIRE(buffered_store.store.values.size() == 8);
  {
    auto gen = buffered_store.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 1, 'a'));
    REQUIRE(gen.next() == std::tuple(1, 3, '0'));
    REQUIRE(gen.next() == std::tuple(3, 4, 'b'));
    REQUIRE(gen.next() == std::tuple(4, 5, '1'));
    REQUIRE(gen.next() == std::tuple(5, 6, '2'));
    REQUIRE(gen.next() == std::tuple(6, 7, 'd'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'e'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'f'));
    REQUIRE(gen.done());
  }

  // Test a degenerate case as well.
  buffered_store.buffer = {
      std::tuple(0, 9, 'X'),
  };
  buffered_store.flush();
  REQUIRE(buffered_store.buffer.empty());
  {
    auto gen = buffered_store.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 9, 'X'));
    REQUIRE(gen.done());
  }
}

TEST_CASE("Test access and assignment of RangeMap", "[range_map]") {
  auto rm = RangeMap<char>::make(10, 'a');

  // Check that the existing range is defined correctly.
  REQUIRE(rm.get(0) == 'a');
  REQUIRE(rm.get(1) == 'a');
  REQUIRE(rm.get(9) == 'a');
}
