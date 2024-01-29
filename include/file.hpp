#ifndef FILE_HPP
#define FILE_HPP

#include <fstream>
#include <sstream>
#include <string>

class File {
   public:
    static auto getContentFromPath(const std::string& filePath) -> std::string;
    auto updateFromPath() { content = getContentFromPath(path); }

    explicit File(std::string path_)
        : path{std::move(path_)},
          content{getContentFromPath(path)},
          rawContent{content.c_str()} {}
    explicit File() = default;

    friend auto operator<<(std::ostream& outputStream, const File& file)
        -> std::ostream& {
        return outputStream << file.content;
    }

    // We basically Java now, sob emoji
    [[nodiscard]] auto getContent() const { return content; }

    [[nodiscard]] auto getRawContent() const {
        return rawContent;
    }

    // this one's for the annoying functions
    // that expect string[],
    // even tho the array could be length 1?
    [[nodiscard]] auto getRawContentP() const {
        return &rawContent;
    }

   private:
    std::string path;
    std::string content;
    const char* rawContent = nullptr;
};

inline auto File::getContentFromPath(const std::string& filePath)
    -> std::string {
    std::ifstream fileStream;

    // ensure ifstream objects can throw exceptions:
    // but like... is this even the best way to go really???
    fileStream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fileStream.open(filePath);

    std::stringstream fileContentStream;
    fileContentStream << fileStream.rdbuf();

    return fileContentStream.str();
}

#endif  // FILE_HPP
