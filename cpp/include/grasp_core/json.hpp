#pragma once

// A JSON reader small enough to vendor. The alternative was nlohmann/json or
// RapidJSON, either of which would have to be fetched at Docker build time or
// committed in full; both are far more library than a hundred lines of
// configuration warrant, and neither may run inside the measured path anyway.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace grasp_core::json
{

class ParseError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

class Value
{
public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Value() = default;

  Type type() const noexcept {return type_;}
  bool is_object() const noexcept {return type_ == Type::Object;}
  bool is_array() const noexcept {return type_ == Type::Array;}

  bool contains(const std::string & key) const;

  // Throws ParseError naming the key rather than returning a default: a
  // configuration file missing a field must abort start-up, not silently
  // measure a different pipeline.
  const Value & operator[](const std::string & key) const;
  const Value & operator[](std::size_t index) const;

  double number() const;
  int integer() const;
  bool boolean() const;
  const std::string & string() const;
  const std::vector<Value> & array() const;

  std::size_t size() const noexcept;

  // Fills exactly n numbers from an array-valued node, or throws.
  void fill(double * out, std::size_t n) const;

private:
  friend class Parser;

  Type type_{Type::Null};
  bool bool_{false};
  double number_{0.0};
  std::string string_;
  std::vector<Value> array_;
  std::vector<std::pair<std::string, Value>> object_;
};

Value parse(const std::string & text);
Value parse_file(const std::string & path);

}  // namespace grasp_core::json
