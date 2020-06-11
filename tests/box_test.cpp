#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <pyskip/detail/box.hpp>

using namespace pyskip::detail::box;

TEST_CASE("Test box put and get of all types", "[box_basic]") {
  REQUIRE(Box(123).get<int>() == 123);
  REQUIRE(Box(-23).get<int>() == -23);
  REQUIRE(Box(123u).get<unsigned int>() == 123u);
  REQUIRE(Box(0xFFFFFFFFu).get<unsigned int>() == 0xFFFFFFFFu);
  REQUIRE(Box('a').get<char>() == 'a');
  REQUIRE(Box(123.456f).get<float>() == 123.456f);

  Box b;
  b = -123;
  b = 123.456f;
  REQUIRE(b.get<float>() == 123.456f);
  b = 'x';
  REQUIRE(b.get<char>() == 'x');
  b = 579u;
  REQUIRE(b.get<unsigned int>() == 579u);
  b = 'r';
  REQUIRE(b.get<char>() == 'r');
}

TEST_CASE("Test collection of boxes", "[box_collection]") {
  std::vector<Box> m;
  for (int i = 0; i < 10; i += 1) {
    m.emplace_back(i % 2 == 0);
  }

  std::vector<Box> x;
  for (char i = 'a'; i < 'a' + 10; i += 1) {
    x.emplace_back(i);
  }

  std::vector<Box> y;
  for (char i = 'A'; i < 'A' + 10; i += 1) {
    y.emplace_back(i);
  }

  std::vector<char> z;
  for (int i = 0; i < 10; i += 1) {
    z.push_back(m[i].get<bool>() ? x[i].get<char>() : y[i].get<char>());
  }

  REQUIRE_THAT(
      z,
      Catch::Equals<char>({'a', 'B', 'c', 'D', 'e', 'F', 'g', 'H', 'i', 'J'}));
}
