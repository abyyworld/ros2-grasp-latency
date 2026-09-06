#include "grasp_core/json.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace grasp_core::json
{

bool Value::contains(const std::string & key) const
{
  if (type_ != Type::Object) {
    return false;
  }
  for (const auto & entry : object_) {
    if (entry.first == key) {
      return true;
    }
  }
  return false;
}

const Value & Value::operator[](const std::string & key) const
{
  if (type_ != Type::Object) {
    throw ParseError("expected an object when looking up '" + key + "'");
  }
  for (const auto & entry : object_) {
    if (entry.first == key) {
      return entry.second;
    }
  }
  throw ParseError("missing key '" + key + "'");
}

const Value & Value::operator[](std::size_t index) const
{
  if (type_ != Type::Array) {
    throw ParseError("expected an array");
  }
  if (index >= array_.size()) {
    throw ParseError("array index out of range");
  }
  return array_[index];
}

double Value::number() const
{
  if (type_ != Type::Number) {
    throw ParseError("expected a number");
  }
  return number_;
}

int Value::integer() const
{
  const double v = number();
  const int i = static_cast<int>(v);
  if (static_cast<double>(i) != v) {
    throw ParseError("expected an integer");
  }
  return i;
}

bool Value::boolean() const
{
  if (type_ != Type::Bool) {
    throw ParseError("expected a boolean");
  }
  return bool_;
}

const std::string & Value::string() const
{
  if (type_ != Type::String) {
    throw ParseError("expected a string");
  }
  return string_;
}

const std::vector<Value> & Value::array() const
{
  if (type_ != Type::Array) {
    throw ParseError("expected an array");
  }
  return array_;
}

std::size_t Value::size() const noexcept
{
  if (type_ == Type::Array) {
    return array_.size();
  }
  if (type_ == Type::Object) {
    return object_.size();
  }
  return 0;
}

void Value::fill(double * out, std::size_t n) const
{
  const auto & items = array();
  if (items.size() != n) {
    throw ParseError(
            "expected " + std::to_string(n) + " numbers, found " +
            std::to_string(items.size()));
  }
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = items[i].number();
  }
}

class Parser
{
public:
  explicit Parser(const std::string & text)
  : text_(text) {}

  Value parse()
  {
    skip_space();
    Value v = parse_value();
    skip_space();
    if (pos_ != text_.size()) {
      fail("trailing content after the top-level value");
    }
    return v;
  }

private:
  [[noreturn]] void fail(const std::string & why) const
  {
    std::size_t line = 1;
    for (std::size_t i = 0; i < pos_ && i < text_.size(); ++i) {
      if (text_[i] == '\n') {
        ++line;
      }
    }
    throw ParseError("JSON line " + std::to_string(line) + ": " + why);
  }

  void skip_space()
  {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else {
        break;
      }
    }
  }

  char peek() const
  {
    if (pos_ >= text_.size()) {
      throw ParseError("unexpected end of input");
    }
    return text_[pos_];
  }

  void expect(char c)
  {
    if (pos_ >= text_.size() || text_[pos_] != c) {
      fail(std::string("expected '") + c + "'");
    }
    ++pos_;
  }

  bool literal(const char * word)
  {
    const std::size_t n = std::char_traits<char>::length(word);
    if (text_.compare(pos_, n, word) == 0) {
      pos_ += n;
      return true;
    }
    return false;
  }

  Value parse_value()
  {
    switch (peek()) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return parse_string_value();
      case 't': case 'f': return parse_bool();
      case 'n': return parse_null();
      default: return parse_number();
    }
  }

  Value parse_object()
  {
    Value v;
    v.type_ = Value::Type::Object;
    expect('{');
    skip_space();
    if (peek() == '}') {
      ++pos_;
      return v;
    }
    for (;;) {
      skip_space();
      std::string key = parse_raw_string();
      skip_space();
      expect(':');
      skip_space();
      v.object_.emplace_back(std::move(key), parse_value());
      skip_space();
      if (peek() == ',') {
        ++pos_;
        continue;
      }
      expect('}');
      return v;
    }
  }

  Value parse_array()
  {
    Value v;
    v.type_ = Value::Type::Array;
    expect('[');
    skip_space();
    if (peek() == ']') {
      ++pos_;
      return v;
    }
    for (;;) {
      skip_space();
      v.array_.push_back(parse_value());
      skip_space();
      if (peek() == ',') {
        ++pos_;
        continue;
      }
      expect(']');
      return v;
    }
  }

  Value parse_bool()
  {
    Value v;
    v.type_ = Value::Type::Bool;
    if (literal("true")) {
      v.bool_ = true;
    } else if (literal("false")) {
      v.bool_ = false;
    } else {
      fail("expected true or false");
    }
    return v;
  }

  Value parse_null()
  {
    if (!literal("null")) {
      fail("expected null");
    }
    return Value{};
  }

  Value parse_string_value()
  {
    Value v;
    v.type_ = Value::Type::String;
    v.string_ = parse_raw_string();
    return v;
  }

  std::string parse_raw_string()
  {
    expect('"');
    std::string out;
    for (;;) {
      if (pos_ >= text_.size()) {
        fail("unterminated string");
      }
      const char c = text_[pos_++];
      if (c == '"') {
        return out;
      }
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (pos_ >= text_.size()) {
        fail("unterminated escape");
      }
      const char e = text_[pos_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u':
          // No configuration value in this repository is non-ASCII, and a
          // half-correct surrogate decoder is worse than an honest refusal.
          fail("\\u escapes are not supported");
          break;
        default: fail("unknown escape"); break;
      }
    }
  }

  Value parse_number()
  {
    const std::size_t start = pos_;
    if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) {
      ++pos_;
    }
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      const bool numeric = (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
        ((c == '-' || c == '+') && (text_[pos_ - 1] == 'e' || text_[pos_ - 1] == 'E'));
      if (!numeric) {
        break;
      }
      ++pos_;
    }
    if (pos_ == start) {
      fail("expected a value");
    }
    const std::string token = text_.substr(start, pos_ - start);
    // strtod rather than stod so a malformed token is caught here instead of
    // through an exception thrown from deep inside the standard library.
    char * end = nullptr;
    const double parsed = std::strtod(token.c_str(), &end);
    if (end != token.c_str() + token.size()) {
      fail("malformed number '" + token + "'");
    }
    Value v;
    v.type_ = Value::Type::Number;
    v.number_ = parsed;
    return v;
  }

  const std::string & text_;
  std::size_t pos_{0};
};

Value parse(const std::string & text)
{
  Parser parser(text);
  return parser.parse();
}

Value parse_file(const std::string & path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw ParseError("cannot open '" + path + "'");
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  try {
    return parse(text);
  } catch (const ParseError & e) {
    throw ParseError(path + ": " + e.what());
  }
}

}  // namespace grasp_core::json
