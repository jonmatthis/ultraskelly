// TooltipIconButton.tsx
import {ComponentPropsWithRef, FC, forwardRef} from 'react';


export interface TooltipIconButtonProps extends ComponentPropsWithRef<'button'> {
    tooltip: string;
    side?: 'top' | 'bottom' | 'left' | 'right';
    variant?: 'default' | 'outline' | 'ghost';
    size?: 'icon' | 'sm' | 'md' | 'lg';
}

export const TooltipIconButton = forwardRef<HTMLButtonElement, TooltipIconButtonProps>(
    ({ children, tooltip, side = 'bottom', className, variant = 'ghost', size = 'icon', ...rest }, ref) => {
        return (
            <TooltipProvider>
                <Tooltip>
                    <TooltipTrigger asChild>
                        <button
                            ref={ref}
                            className={cn(
                                'aui-button',
                                variant && `aui-button-${variant}`,
                                size && `aui-button-${size}`,
                                className
                            )}
                            {...rest}
                        >
                            {children}
                            <span className="sr-only">{tooltip}</span>
                        </button>
                    </TooltipTrigger>
                    <TooltipContent side={side} className="aui-tooltip-content">
                        {tooltip}
                    </TooltipContent>
                </Tooltip>
            </TooltipProvider>
        );
    }
);

TooltipIconButton.displayName = 'TooltipIconButton';

// ----------------------------------------

// MarkdownText.tsx
import { memo } from 'react';
import {
    MarkdownTextPrimitive,
    type CodeHeaderProps,
    unstable_memoizeMarkdownComponents as memoizeMarkdownComponents,
    useIsMarkdownCodeBlock,
} from '@assistant-ui/react-markdown';
import remarkGfm from 'remark-gfm';
import { useState } from 'react';
import { CheckIcon, CopyIcon } from 'lucide-react';

const CodeHeader: FC<CodeHeaderProps> = ({ language, code }) => {
    const { isCopied, copyToClipboard } = useCopyToClipboard();

    const onCopy = () => {
        if (!code || isCopied) return;
        copyToClipboard(code);
    };

    return (
        <div className="aui-code-header-root">
            <span className="aui-code-header-language">{language}</span>
            <TooltipIconButton tooltip="Copy" onClick={onCopy}>
                {!isCopied && <CopyIcon />}
                {isCopied && <CheckIcon />}
            </TooltipIconButton>
        </div>
    );
};

const useCopyToClipboard = ({ copiedDuration = 3000 } = {}) => {
    const [isCopied, setIsCopied] = useState<boolean>(false);

    const copyToClipboard = (value: string) => {
        if (!value) return;
        
        navigator.clipboard.writeText(value).then(() => {
            setIsCopied(true);
            setTimeout(() => setIsCopied(false), copiedDuration);
        });
    };

    return { isCopied, copyToClipboard };
};

const defaultComponents = memoizeMarkdownComponents({
    h1: ({ className, ...props }) => (
        <h1 className={cn('aui-md-h1', className)} {...props} />
    ),
    h2: ({ className, ...props }) => (
        <h2 className={cn('aui-md-h2', className)} {...props} />
    ),
    h3: ({ className, ...props }) => (
        <h3 className={cn('aui-md-h3', className)} {...props} />
    ),
    p: ({ className, ...props }) => (
        <p className={cn('aui-md-p', className)} {...props} />
    ),
    a: ({ className, ...props }) => (
        <a className={cn('aui-md-a', className)} {...props} />
    ),
    blockquote: ({ className, ...props }) => (
        <blockquote className={cn('aui-md-blockquote', className)} {...props} />
    ),
    ul: ({ className, ...props }) => (
        <ul className={cn('aui-md-ul', className)} {...props} />
    ),
    ol: ({ className, ...props }) => (
        <ol className={cn('aui-md-ol', className)} {...props} />
    ),
    hr: ({ className, ...props }) => (
        <hr className={cn('aui-md-hr', className)} {...props} />
    ),
    table: ({ className, ...props }) => (
        <table className={cn('aui-md-table', className)} {...props} />
    ),
    th: ({ className, ...props }) => (
        <th className={cn('aui-md-th', className)} {...props} />
    ),
    td: ({ className, ...props }) => (
        <td className={cn('aui-md-td', className)} {...props} />
    ),
    code: function Code({ className, ...props }) {
        const isCodeBlock = useIsMarkdownCodeBlock();
        return (
            <code
                className={cn(!isCodeBlock && 'aui-md-inline-code', className)}
                {...props}
            />
        );
    },
    CodeHeader,
});

const MarkdownTextImpl = () => {
    return (
        <MarkdownTextPrimitive
            remarkPlugins={[remarkGfm]}
            className="aui-md"
            components={defaultComponents}
        />
    );
};

export const MarkdownText = memo(MarkdownTextImpl);

// ----------------------------------------

// cn.ts - Utility for merging class names
import { type ClassValue, clsx } from 'clsx';
import {Tooltip, TooltipContent, TooltipProvider, TooltipTrigger} from "@radix-ui/react-tooltip";

export function cn(...inputs: ClassValue[]) {
    return clsx(inputs);
}