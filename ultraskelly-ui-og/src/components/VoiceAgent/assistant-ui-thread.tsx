import {
    ThreadPrimitive,
    MessagePrimitive,
    ComposerPrimitive,
    BranchPickerPrimitive,
    ActionBarPrimitive,
} from '@assistant-ui/react';
import {
    ArrowDownIcon,
    ArrowUpIcon,
    CheckIcon,
    ChevronLeftIcon,
    ChevronRightIcon,
    CopyIcon,
    PencilIcon,
    RefreshCwIcon,
    Square,
} from 'lucide-react';
import { type FC } from 'react';
import {cn, MarkdownText, TooltipIconButton} from "./assistant-ui-supporting-components.tsx";

export const Thread: FC = () => {
    return (
        <ThreadPrimitive.Root className="aui-root aui-thread-root">
            <ThreadPrimitive.Viewport className="aui-thread-viewport">
                <ThreadPrimitive.If empty>
                    <ThreadWelcome />
                </ThreadPrimitive.If>

                <ThreadPrimitive.Messages
                    components={{
                        UserMessage,
                        EditComposer,
                        AssistantMessage,
                    }}
                />

                <ThreadPrimitive.If empty={false}>
                    <div className="aui-thread-viewport-spacer" />
                </ThreadPrimitive.If>

                <Composer />
            </ThreadPrimitive.Viewport>
        </ThreadPrimitive.Root>
    );
};

const ThreadWelcome: FC = () => {
    return (
        <div className="aui-thread-welcome-root">
            <div className="aui-thread-welcome-center">
                <div className="aui-thread-welcome-message">
                    <div className="aui-thread-welcome-message-motion-1">
                        🎤 Voice Agent Ready
                    </div>
                    <div className="aui-thread-welcome-message-motion-2">
                        Speak or type your message
                    </div>
                </div>
            </div>
        </div>
    );
};

const ThreadScrollToBottom: FC = () => {
    return (
        <ThreadPrimitive.ScrollToBottom asChild>
            <TooltipIconButton
                tooltip="Scroll to bottom"
                variant="outline"
                className="aui-thread-scroll-to-bottom"
            >
                <ArrowDownIcon />
            </TooltipIconButton>
        </ThreadPrimitive.ScrollToBottom>
    );
};

const Composer: FC = () => {
    return (
        <div className="aui-composer-wrapper">
            <ThreadScrollToBottom />
            <ComposerPrimitive.Root className="aui-composer-root">
                <ComposerPrimitive.Input
                    placeholder="Type a message..."
                    className="aui-composer-input"
                    rows={1}
                    autoFocus={false}
                    aria-label="Message input"
                />
                <ComposerAction />
            </ComposerPrimitive.Root>
        </div>
    );
};

const ComposerAction: FC = () => {
    return (
        <div className="aui-composer-action-wrapper">
            <ThreadPrimitive.If running={false}>
                <ComposerPrimitive.Send asChild>
                    <TooltipIconButton
                        tooltip="Send message"
                        side="bottom"
                        variant="default"
                        size="icon"
                        className="aui-composer-send"
                        aria-label="Send message"
                    >
                        <ArrowUpIcon className="aui-composer-send-icon" />
                    </TooltipIconButton>
                </ComposerPrimitive.Send>
            </ThreadPrimitive.If>

            <ThreadPrimitive.If running>
                <ComposerPrimitive.Cancel asChild>
                    <button
                        type="button"
                        className="aui-composer-cancel"
                        aria-label="Stop generating"
                    >
                        <Square className="aui-composer-cancel-icon" />
                    </button>
                </ComposerPrimitive.Cancel>
            </ThreadPrimitive.If>
        </div>
    );
};

const AssistantMessage: FC = () => {
    return (
        <MessagePrimitive.Root className="aui-assistant-message-root">
            <div className="aui-assistant-message-content">
                <MessagePrimitive.Parts
                    components={{
                        Text: MarkdownText,
                    }}
                />
            </div>
            <div className="aui-assistant-message-footer">
                <BranchPicker />
                <AssistantActionBar />
            </div>
        </MessagePrimitive.Root>
    );
};

const AssistantActionBar: FC = () => {
    return (
        <ActionBarPrimitive.Root
            hideWhenRunning
            autohide="not-last"
            autohideFloat="single-branch"
            className="aui-assistant-action-bar-root"
        >
            <ActionBarPrimitive.Copy asChild>
                <TooltipIconButton tooltip="Copy">
                    <MessagePrimitive.If copied>
                        <CheckIcon />
                    </MessagePrimitive.If>
                    <MessagePrimitive.If copied={false}>
                        <CopyIcon />
                    </MessagePrimitive.If>
                </TooltipIconButton>
            </ActionBarPrimitive.Copy>

            <ActionBarPrimitive.Reload asChild>
                <TooltipIconButton tooltip="Refresh">
                    <RefreshCwIcon />
                </TooltipIconButton>
            </ActionBarPrimitive.Reload>
        </ActionBarPrimitive.Root>
    );
};

const UserMessage: FC = () => {
    return (
        <MessagePrimitive.Root className="aui-user-message-root">
            <div className="aui-user-message-content-wrapper">
                <div className="aui-user-message-content">
                    <MessagePrimitive.Parts />
                </div>
                <div className="aui-user-action-bar-wrapper">
                    <UserActionBar />
                </div>
            </div>
            <BranchPicker className="aui-user-branch-picker" />
        </MessagePrimitive.Root>
    );
};

const UserActionBar: FC = () => {
    return (
        <ActionBarPrimitive.Root
            hideWhenRunning
            autohide="not-last"
            className="aui-user-action-bar-root"
        >
            <ActionBarPrimitive.Edit asChild>
                <TooltipIconButton tooltip="Edit">
                    <PencilIcon />
                </TooltipIconButton>
            </ActionBarPrimitive.Edit>
        </ActionBarPrimitive.Root>
    );
};

const EditComposer: FC = () => {
    return (
        <div className="aui-edit-composer-wrapper">
            <ComposerPrimitive.Root className="aui-edit-composer-root">
                <ComposerPrimitive.Input
                    className="aui-edit-composer-input"
                    autoFocus
                />
                <div className="aui-edit-composer-footer">
                    <ComposerPrimitive.Cancel asChild>
                        <button className="aui-edit-composer-cancel">
                            Cancel
                        </button>
                    </ComposerPrimitive.Cancel>
                    <ComposerPrimitive.Send asChild>
                        <button className="aui-edit-composer-send">
                            Update
                        </button>
                    </ComposerPrimitive.Send>
                </div>
            </ComposerPrimitive.Root>
        </div>
    );
};

const BranchPicker: FC<{ className?: string }> = ({ className }) => {
    return (
        <BranchPickerPrimitive.Root
            hideWhenSingleBranch
            className={cn('aui-branch-picker-root', className)}
        >
            <BranchPickerPrimitive.Previous asChild>
                <TooltipIconButton tooltip="Previous">
                    <ChevronLeftIcon />
                </TooltipIconButton>
            </BranchPickerPrimitive.Previous>

            <span className="aui-branch-picker-state">
                <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
            </span>

            <BranchPickerPrimitive.Next asChild>
                <TooltipIconButton tooltip="Next">
                    <ChevronRightIcon />
                </TooltipIconButton>
            </BranchPickerPrimitive.Next>
        </BranchPickerPrimitive.Root>
    );
};
