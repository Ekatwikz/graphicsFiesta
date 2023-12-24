#!/usr/bin/env sh

USAGE="Usage: $0 [-j] [-c [format|lint]] [-a [format|lint|all]]"

STARTDIR=$(dirname "$0")
FORMAT_PATHS="$(echo "$STARTDIR"/lib/*.cpp) $(echo "$STARTDIR"/include/*.hpp) $(echo "$STARTDIR"/src/*.cpp)"
CFLAGS="-I$STARTDIR/include -I$STARTDIR/ -std=c++20"
RET=0
main() {
	if [ "$1" = "format" ]; then
		assert_exists "clang-format"

		# shellcheck disable=SC2086
		EVAL_STR="clang-format $FORMAT_ARGS"
		parallel_or_eval "$EVAL_STR" "$EVAL_STR $FORMAT_PATHS" && say_ok
		RET=$((RET | $?))
	elif [ "$1" = "lint" ]; then
		assert_exists "clang-tidy"

		parallel_or_eval \
			"clang-tidy $TIDY_ARGS {} -- $CFLAGS" \
			"clang-tidy $TIDY_ARGS $FORMAT_PATHS -- $CFLAGS" \
			&& say_ok
		RET=$((RET | $?))
	elif [ "$1" = "all" ]; then
		main "format"
		echo
		main "lint"
	fi
}

parallel_or_eval() {
	if [ -n "$PARALLEL" ]; then
		# NB: make sure to NOT use the moreutils version!
		assert_exists "parallel"

		# shellcheck disable=SC2086
		parallel "$1" ::: $FORMAT_PATHS
	else
		eval "$2"
	fi
}

check_opts() {
	[ $# -eq 0 ] && fail_msg "$USAGE"

	while getopts ":c:a:jh" opt; do
		case "${opt}" in
			c)
				case "$OPTARG" in
					format|lint)
						FORMAT_ARGS="--verbose --dry-run --Werror"
						TIDY_ARGS=""
						ACTION="$OPTARG"
						;;
					*)
						fail_msg "$USAGE"
						;;
				esac
				;;
			a)
				case "$OPTARG" in
					format|lint|all)
						FORMAT_ARGS="--verbose -i"
						TIDY_ARGS="--fix"
						ACTION="$OPTARG"
						;;
					*)
						fail_msg "$USAGE"
						;;
				esac
				;;
			h)
				printf "%s\n" "$USAGE" && exit ;;
			j)
				PARALLEL=1
				;;
			*)
				fail_msg "$USAGE" ;;
		esac

		main "$ACTION"
	done

	shift $((OPTIND-1)) # unnecessary?
	exit $RET
}

assert_exists() {
	command -v "$1" >/dev/null \
		|| fail_msg "No $1??"
}

fail_msg() {
	printf "%s\n" "$1" >&2
	exit 1
}

say_ok() {
	printf "\n\033[0;30;42mOKAY!\033[0m\n"
}

check_opts "$@"
