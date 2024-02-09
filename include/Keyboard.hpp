// Blatantly stolen code (from myself: https://github.com/Ekatwikz/Basic-Arkanoid/blob/main/gameSrc/Keyboard.h)

// This is a little state machine that can be used to figure out
// which key is effectively pressed out of a
// mutually exclusive pair (eg: Left-Right, Up-Down, etc)
// if you only have Press and Release events

// So you can do stuff like: LeftPress => Left is active
// or: LeftPress,RightPress,LeftRelease,RightRelease => [Nothing] is active
// but also: LeftPress,RightPress,RightRelease => Left is active
// and: LeftPress,RightPress => Right is active
// (etc)

// also nice if you do have Press and Hold events,
// but the Hold event has a delay,
// which can make simply setting a handler for "Press or Hold" feel unacceptable

#ifndef KEYBOARD_H
#define KEYBOARD_H

#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

class KeyPair {
private:
	// TODO: std::array instead ig?
	int keyStates[2]{}, direction = 0;

public:
	enum PossibleKeyState { UNPRESSED, PRESSED, STANDBY = -1 };
	enum Direction { FIRST = -1, NONE, SECOND };

	void setPressed(bool pos, bool state) {
		// switch state of other position
		int* other = &keyStates[!pos];
		if ((state == PRESSED && *other == PRESSED)
				|| (state == UNPRESSED && *other == STANDBY)) {
			*other *= -1;
		}

		// set state of given position
		keyStates[pos] = state;

		// update direction
		direction = 0;
		if (keyStates[0] == PRESSED) {
			direction = -1;
		} else if (keyStates[1] == PRESSED) {
			direction = 1;
		}
	}

	[[ nodiscard ]] auto getDirection() const -> Direction {
		return static_cast<Direction>(direction);
	}
};

class KeyboardState {
private:
	KeyPair leftRight, upDown, forwardBackward;
	enum { KB_LEFT, KB_RIGHT };
	enum { KB_UP, KB_DOWN };
	enum { KB_FORWARD, KB_BACKWARD };

	KeyPair alpha, beta, gamma;
	enum { KB_ALPHA_DOWN, KB_ALPHA_UP };
	enum { KB_BETA_DOWN, KB_BETA_UP };
	enum { KB_GAMMA_DOWN, KB_GAMMA_UP };

public: 
	void handleKeyPress(int key, bool isPressed) {
		// NB: the texture is flipped,
		// so there are some boonk shenanigans to handle that here

		switch (key) {
			case GLFW_KEY_A:
				leftRight.setPressed(KB_LEFT, isPressed);
				break;
			case GLFW_KEY_D:
				leftRight.setPressed(KB_RIGHT, isPressed);
				break;
			case GLFW_KEY_E:
				upDown.setPressed(KB_UP, isPressed);
				break;
			case GLFW_KEY_Q:
				upDown.setPressed(KB_DOWN, isPressed);
				break;
			case GLFW_KEY_W:
				forwardBackward.setPressed(KB_FORWARD, isPressed);
				break;
			case GLFW_KEY_S:
				forwardBackward.setPressed(KB_BACKWARD, isPressed);
				break;

			case GLFW_KEY_J:
				gamma.setPressed(KB_GAMMA_UP, isPressed);
				break;
			case GLFW_KEY_L:
				gamma.setPressed(KB_GAMMA_DOWN, isPressed);
				break;
			case GLFW_KEY_I:
				alpha.setPressed(KB_ALPHA_DOWN, isPressed);
				break;
			case GLFW_KEY_K:
				alpha.setPressed(KB_ALPHA_UP, isPressed);
				break;
			case GLFW_KEY_U:
				beta.setPressed(KB_BETA_DOWN, isPressed);
				break;
			case GLFW_KEY_O:
				beta.setPressed(KB_BETA_UP, isPressed);
				break;

			default: // ??
				break;
		}
	}

	[[ nodiscard ]] auto getPositionDelta() const -> float3 {
		return { leftRight.getDirection(), upDown.getDirection(), forwardBackward.getDirection() };
	}

	[[ nodiscard ]] auto getEulerDelta() const -> float3 {
		return { alpha.getDirection(), gamma.getDirection(), beta.getDirection() };
	}
};

#endif // KEYBOARD_H
